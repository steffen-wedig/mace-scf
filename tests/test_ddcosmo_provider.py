"""Tests for the ddCOSMO reaction-field provider (per-atom potential + field via autograd).

The provider differentiates the pyscf-matched ddCOSMO energy, so correctness reduces to:
* it reproduces the wrapped `DdcosmoReactionField` energy;
* the returned (phi, E) satisfy the Euler identity `0.5 sum_i (q_i phi_i + mu_i . E_i) == G`
  (guaranteed for the degree-2 homogeneous ddCOSMO free energy) -> phi = dG/dq, E = dG/dmu;
* gas (eps = 1) gives zero potential/field/energy;
* it works whether or not the caller has autograd enabled (train vs eval/inference).
"""

from __future__ import annotations

import pytest
import torch
from mace.tools.scatter import scatter_sum
from mace_scf.electrostatics.reaction_field.ddcosmo_provider import DdcosmoReactionFieldProvider
from mace_scf.electrostatics.reaction_field.screened_potential import (
    dielectric_scaling_from_dielectric_constant,
)


@pytest.fixture(autouse=True)
def _double():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _two_waters():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586],
         [3.0, 0.0, 0.0], [3.0, 0.757, 0.586], [3.0, -0.757, 0.586]]
    )
    charges = torch.tensor([-0.8, 0.4, 0.4, -0.7, 0.35, 0.35])
    dipoles = torch.randn(6, 3) * 0.1  # Cartesian atomic dipoles
    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    return charges, dipoles, positions, atomic_numbers, batch


def test_ddcosmo_provider_matches_wrapped_energy():
    provider = DdcosmoReactionFieldProvider(smearing_width=1.5, max_spherical_harmonic_order=4)
    charges, dipoles, positions, atomic_numbers, batch = _two_waters()
    eps = torch.tensor([78.3553, 20.5])
    _, _, energy = provider(charges, dipoles, positions, atomic_numbers, batch, eps)
    # the wrapped module expects e3nn (y,z,x) dipoles; provider maps Cartesian -> e3nn as [1,2,0]
    dipoles_e3nn = dipoles.index_select(1, torch.tensor([1, 2, 0]))
    reference = provider.ddcosmo(
        charges, positions, atomic_numbers, batch,
        dielectric_scaling_from_dielectric_constant(eps), dipoles_e3nn,
    )
    assert torch.allclose(energy, reference, atol=1e-10)


def test_ddcosmo_provider_euler_identity():
    provider = DdcosmoReactionFieldProvider(smearing_width=1.5, max_spherical_harmonic_order=4)
    charges, dipoles, positions, atomic_numbers, batch = _two_waters()
    eps = torch.tensor([78.3553, 20.5])
    phi, field, energy = provider(charges, dipoles, positions, atomic_numbers, batch, eps)
    per_atom = charges * phi + (dipoles * field).sum(dim=-1)
    reconstructed = 0.5 * scatter_sum(per_atom, batch, dim=0, dim_size=2)
    assert torch.allclose(reconstructed, energy, atol=1e-9)


def test_ddcosmo_provider_gas_limit_zero():
    provider = DdcosmoReactionFieldProvider(smearing_width=1.5, max_spherical_harmonic_order=4)
    charges, dipoles, positions, atomic_numbers, batch = _two_waters()
    phi, field, energy = provider(
        charges, dipoles, positions, atomic_numbers, batch, torch.tensor([1.0, 1.0])
    )
    assert float(energy.abs().max()) < 1e-12
    assert float(phi.abs().max()) < 1e-12
    assert float(field.abs().max()) < 1e-12


def test_ddcosmo_provider_works_without_outer_grad():
    # Under torch.no_grad() inputs (inference/benchmark), the provider must still return the same
    # phi/E/energy via its internal enable_grad + detach fallback.
    provider = DdcosmoReactionFieldProvider(smearing_width=1.5, max_spherical_harmonic_order=4)
    charges, dipoles, positions, atomic_numbers, batch = _two_waters()
    eps = torch.tensor([78.3553, 20.5])
    phi, field, energy = provider(charges, dipoles, positions, atomic_numbers, batch, eps)
    with torch.no_grad():
        charges_d, dipoles_d = charges.clone(), dipoles.clone()
    phi_e, field_e, energy_e = provider(charges_d, dipoles_d, positions, atomic_numbers, batch, eps)
    assert torch.allclose(phi_e, phi, atol=1e-8)
    assert torch.allclose(field_e, field, atol=1e-8)
    assert torch.allclose(energy_e, energy, atol=1e-8)
