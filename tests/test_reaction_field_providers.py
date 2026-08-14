"""Tests for the reaction-field *providers* (per-atom potential + field, for in-loop injection).

Two tiers share the :class:`ReactionFieldProvider` interface:

* :class:`GeneralizedBornReactionField` -- monopole; validated to reproduce the existing
  :class:`ScreenedGeneralizedBornSolvation` energy exactly and to carry no dipole field.
* :class:`GeneralizedKirkwoodReactionField` -- charge + atomic dipole; validated against the
  analytic Born (monopole) and Onsager (dipole) self-energies, the monopole limit (which reduces
  to unscreened Still-GB at ``gkc = 4``), and the internal symmetry ``phi = dG/dq``, ``E = dG/dmu``
  which guarantees the charge-dipole / dipole-dipole tensors form a consistent energy.
"""

from __future__ import annotations

import pytest
import torch
from mace_scf.electrostatics.reaction_field import dielectric_scaling
from mace_scf.electrostatics.reaction_field.generalized_kirkwood import (
    GeneralizedKirkwoodReactionField,
)
from mace_scf.electrostatics.reaction_field.reaction_potential import (
    GeneralizedBornReactionField,
)
from mace_scf.electrostatics.reaction_field.screened_born import (
    ScreenedGeneralizedBornSolvation,
)
from mace_scf.electrostatics.reaction_field.screened_potential import (
    COULOMB_CONSTANT_EV_ANGSTROM,
    build_cavity_radius_lookup,
    compute_obc_born_radii,
    kirkwood_dielectric_factor,
)

WATER_POSITIONS = [
    [0.0, 0.0, 0.117],
    [0.0, 0.757, -0.469],
    [0.0, -0.757, -0.469],
]
WATER_CHARGES = [-0.8, 0.4, 0.4]
WATER_ATOMIC_NUMBERS = [8, 1, 1]
WATER = 78.3553


@pytest.fixture(autouse=True)
def _use_double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _water(dipoles=None):
    positions = torch.tensor(WATER_POSITIONS)
    charges = torch.tensor(WATER_CHARGES)
    atomic_numbers = torch.tensor(WATER_ATOMIC_NUMBERS, dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    if dipoles is None:
        dipoles = torch.zeros(3, 3)
    return charges, dipoles, positions, atomic_numbers, batch


# --------------------------------------------------------------------------------------------
# Monopole generalized-Born provider
# --------------------------------------------------------------------------------------------


def test_gb_provider_energy_matches_screened_born():
    provider = GeneralizedBornReactionField(smearing_width=1.5)
    legacy = ScreenedGeneralizedBornSolvation(smearing_width=1.5)
    charges, dipoles, positions, atomic_numbers, batch = _water()
    eps = torch.tensor([WATER])
    _, field, energy = provider(charges, dipoles, positions, atomic_numbers, batch, eps)
    legacy_energy = legacy(
        charges, positions, atomic_numbers, batch, torch.tensor([dielectric_scaling(WATER)])
    )
    assert torch.allclose(energy, legacy_energy, atol=1e-12)
    # a pure Born (monopole) model produces no field conjugate to the atomic dipole.
    assert torch.count_nonzero(field) == 0


def test_gb_provider_gas_and_consistency():
    provider = GeneralizedBornReactionField(smearing_width=1.5)
    charges, dipoles, positions, atomic_numbers, batch = _water()
    _, _, gas_energy = provider(
        charges, dipoles, positions, atomic_numbers, batch, torch.tensor([1.0])
    )
    assert gas_energy.item() == 0.0

    charges_grad = charges.clone().requires_grad_(True)
    potential, _, energy = provider(
        charges_grad, dipoles, positions, atomic_numbers, batch, torch.tensor([WATER])
    )
    (grad,) = torch.autograd.grad(energy.sum(), charges_grad)
    assert torch.allclose(grad, potential, atol=1e-10)  # phi = dG/dq


def test_gb_provider_torchscript():
    provider = GeneralizedBornReactionField(smearing_width=1.5)
    scripted = torch.jit.script(provider)
    charges, dipoles, positions, atomic_numbers, batch = _water()
    eps = torch.tensor([WATER])
    eager = provider(charges, dipoles, positions, atomic_numbers, batch, eps)
    script = scripted(charges, dipoles, positions, atomic_numbers, batch, eps)
    for a, b in zip(eager, script, strict=True):
        assert torch.allclose(a, b, atol=1e-12)


# --------------------------------------------------------------------------------------------
# Generalized-Kirkwood provider
# --------------------------------------------------------------------------------------------


def test_gk_dielectric_factors():
    eps = torch.tensor([WATER])
    assert torch.allclose(kirkwood_dielectric_factor(eps, 0), (eps - 1) / eps)
    assert torch.allclose(kirkwood_dielectric_factor(eps, 1), 2 * (eps - 1) / (2 * eps + 1))
    # gas: every order vanishes.
    for order in (0, 1, 2):
        assert kirkwood_dielectric_factor(torch.tensor([1.0]), order).item() == 0.0


def test_gk_single_atom_born_and_onsager_self_energy():
    provider = GeneralizedKirkwoodReactionField(smearing_width=1.5)
    positions = torch.zeros(1, 3)
    charge = torch.tensor([1.0])
    dipole = torch.tensor([[0.3, -0.2, 0.1]])
    atomic_numbers = torch.tensor([8], dtype=torch.long)
    batch = torch.zeros(1, dtype=torch.long)
    eps = torch.tensor([WATER])
    _, _, energy = provider(charge, dipole, positions, atomic_numbers, batch, eps)

    born = compute_obc_born_radii(
        positions, atomic_numbers, batch, build_cavity_radius_lookup()
    ).item()
    f0 = kirkwood_dielectric_factor(eps, 0).item()
    f1 = kirkwood_dielectric_factor(eps, 1).item()
    analytic = -0.5 * COULOMB_CONSTANT_EV_ANGSTROM * (
        f0 * charge.item() ** 2 / born + f1 * float((dipole**2).sum()) / born**3
    )
    assert abs(energy.item() - analytic) < 1e-10


def test_gk_monopole_limit_matches_unscreened_gb():
    # With dipoles zero and the Still constant c = 4, GK reduces to the unscreened point-charge
    # generalized Born (which GeneralizedBornReactionField recovers as its smearing width -> 0).
    gk = GeneralizedKirkwoodReactionField(smearing_width=1.5, gaussian_constant=4.0)
    gb_unscreened = GeneralizedBornReactionField(smearing_width=1e-4)
    charges, dipoles, positions, atomic_numbers, batch = _water()
    eps = torch.tensor([WATER])
    _, _, gk_energy = gk(charges, dipoles, positions, atomic_numbers, batch, eps)
    _, _, gb_energy = gb_unscreened(charges, dipoles, positions, atomic_numbers, batch, eps)
    assert abs(gk_energy.item() - gb_energy.item()) < 1e-6


def test_gk_gas_limit_zero():
    provider = GeneralizedKirkwoodReactionField(smearing_width=1.5)
    charges, _, positions, atomic_numbers, batch = _water()
    dipoles = torch.randn(3, 3)
    potential, field, energy = provider(
        charges, dipoles, positions, atomic_numbers, batch, torch.tensor([1.0])
    )
    assert energy.item() == 0.0
    assert torch.count_nonzero(potential) == 0
    assert torch.count_nonzero(field) == 0


def test_gk_potential_field_are_energy_gradients():
    # phi = dG/dq and E = dG/dmu -> the charge-dipole and dipole-dipole tensors are symmetric
    # and the injected fields are consistent with the returned energy.
    provider = GeneralizedKirkwoodReactionField(smearing_width=1.5)
    charges, _, positions, atomic_numbers, batch = _water()
    charges = charges.clone().requires_grad_(True)
    dipoles = torch.randn(3, 3, requires_grad=True)
    eps = torch.tensor([WATER])
    potential, field, energy = provider(
        charges, dipoles, positions, atomic_numbers, batch, eps
    )
    (grad_q,) = torch.autograd.grad(energy.sum(), charges, retain_graph=True)
    (grad_mu,) = torch.autograd.grad(energy.sum(), dipoles)
    assert torch.allclose(grad_q, potential, atol=1e-9)
    assert torch.allclose(grad_mu, field, atol=1e-9)


def test_gk_dipoles_drive_nonzero_field():
    # The whole point of GK over GB: a charge distribution produces a field that couples to
    # dipoles (nonzero E), so the solvent can re-polarise the atomic dipoles.
    provider = GeneralizedKirkwoodReactionField(smearing_width=1.5)
    charges, dipoles, positions, atomic_numbers, batch = _water()
    _, field, _ = provider(
        charges, dipoles, positions, atomic_numbers, batch, torch.tensor([WATER])
    )
    assert torch.count_nonzero(field) > 0


def test_gk_force_finite_difference():
    provider = GeneralizedKirkwoodReactionField(smearing_width=1.5)
    charges, _, positions, atomic_numbers, batch = _water()
    dipoles = torch.randn(3, 3)
    eps = torch.tensor([WATER])

    positions_grad = positions.clone().requires_grad_(True)
    energy = provider(charges, dipoles, positions_grad, atomic_numbers, batch, eps)[2].sum()
    (analytic,) = torch.autograd.grad(energy, positions_grad)

    step = 1e-6
    finite = torch.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for component in range(3):
            plus = positions.clone()
            plus[atom, component] += step
            minus = positions.clone()
            minus[atom, component] -= step
            energy_plus = provider(charges, dipoles, plus, atomic_numbers, batch, eps)[2].sum()
            energy_minus = provider(charges, dipoles, minus, atomic_numbers, batch, eps)[2].sum()
            finite[atom, component] = (energy_plus - energy_minus) / (2 * step)
    assert torch.allclose(analytic, finite, atol=1e-6)


def test_gk_batch_isolation():
    provider = GeneralizedKirkwoodReactionField(smearing_width=1.5)
    charges, _, positions, atomic_numbers, batch = _water()
    dipoles = torch.randn(3, 3)
    eps_a, eps_b = WATER, 20.5

    offset = torch.tensor([4.0, -3.0, 2.0])
    batched_positions = torch.cat([positions, positions + offset], dim=0)
    batched_charges = torch.cat([charges, charges])
    batched_dipoles = torch.cat([dipoles, dipoles])
    batched_z = torch.cat([atomic_numbers, atomic_numbers])
    batched_batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    _, _, batched_energy = provider(
        batched_charges,
        batched_dipoles,
        batched_positions,
        batched_z,
        batched_batch,
        torch.tensor([eps_a, eps_b]),
    )
    _, _, energy_a = provider(
        charges, dipoles, positions, atomic_numbers, batch, torch.tensor([eps_a])
    )
    _, _, energy_b = provider(
        charges, dipoles, positions, atomic_numbers, batch, torch.tensor([eps_b])
    )
    assert torch.allclose(batched_energy[0], energy_a[0], atol=1e-11)
    assert torch.allclose(batched_energy[1], energy_b[0], atol=1e-11)


def test_gk_torchscript():
    provider = GeneralizedKirkwoodReactionField(smearing_width=1.5)
    scripted = torch.jit.script(provider)
    charges, _, positions, atomic_numbers, batch = _water()
    dipoles = torch.randn(3, 3)
    eps = torch.tensor([WATER])
    eager = provider(charges, dipoles, positions, atomic_numbers, batch, eps)
    script = scripted(charges, dipoles, positions, atomic_numbers, batch, eps)
    for a, b in zip(eager, script, strict=True):
        assert torch.allclose(a, b, atol=1e-12)


def test_gk_dipole_dipole_large_separation_tensor():
    # At large separation f -> r and the descreening factor -> 1, so the dipole-dipole reaction
    # tensor -> -k f_1 (delta - 3 r^ r^)/r^3 (the standard dipole interaction shape). Check the
    # field a unit dipole at j induces at i for a far, non-self pair.
    provider = GeneralizedKirkwoodReactionField(smearing_width=1.5, gaussian_constant=4.0)
    separation = 40.0
    positions = torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    charges = torch.zeros(2)
    dipoles = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # dipole on atom j = 1 along x
    atomic_numbers = torch.tensor([8, 8], dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)
    eps = torch.tensor([WATER])
    _, field, _ = provider(charges, dipoles, positions, atomic_numbers, batch, eps)

    f1 = kirkwood_dielectric_factor(eps, 1).item()
    # r_ij for i=0 is +x; (delta - 3 r^r^) applied to mu=(1,0,0) gives (1-3, 0, 0) = (-2,0,0).
    expected_x = -COULOMB_CONSTANT_EV_ANGSTROM * f1 * (-2.0) / separation**3
    assert abs(field[0, 0].item() - expected_x) < 1e-6
    assert abs(field[0, 1].item()) < 1e-9
    assert abs(field[0, 2].item()) < 1e-9
