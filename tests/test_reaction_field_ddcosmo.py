"""Tests for the ddCOSMO reaction-field production module.

Independent-solver control: pyscf IS installed and has ``pyscf.solvent.ddcosmo``. The module is
matched against pyscf's own ddCOSMO building blocks (``make_grids_one_sphere``, ``build`` ->
``r_vdw / ylm_1sph / ui / Lmat``) driven with a FIXED set of atomic *point* charges, with the
same cavity radii, Lebedev order, ``lmax`` and ``eta``. The comparison basis is:

* Cavity: identical radii ``1.2 * modified_Bondi[Z]`` (fed to pyscf via ``radii_table`` in Bohr).
* Solute: point charges at the atom centres; the module is driven in its point-charge limit
  (tiny ``smearing_width``) so its erf-screened Gaussian potential collapses onto pyscf's
  point-charge nuclear potential ``sum_a q_a * erf(xi r)/r`` -> ``sum_a q_a / r``.
* Energy: the module works in eV / Angstrom with the Coulomb constant; pyscf works in
  Hartree / Bohr. The two are related by ``E_eV = E_Hartree * HARTREE_IN_ELECTRON_VOLT``
  (equivalently the Coulomb constant equals ``BOHR_IN_ANGSTROM * HARTREE_IN_ELECTRON_VOLT``).

Agreement is to ~1e-8 relative (limited only by the last digits of the units constants). The
physics controls (correct sign, exact ``f(eps)`` scaling across the ladder, monotonic
strengthening with eps) are checked at the design default ``smearing_width`` too.
"""

from __future__ import annotations

import numpy
import pytest
import torch
from mace_scf.electrostatics.reaction_field import (
    HARTREE_IN_ELECTRON_VOLT,
    DdcosmoReactionField,
    dielectric_scaling,
)
from mace_scf.electrostatics.reaction_field.constants import BOHR_IN_ANGSTROM


@pytest.fixture(autouse=True)
def _use_double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


WATER_POSITIONS = [
    [0.0, 0.0, 0.117],
    [0.0, 0.757, -0.469],
    [0.0, -0.757, -0.469],
]
WATER_CHARGES = [-0.8, 0.4, 0.4]
WATER_ATOMIC_NUMBERS = [8, 1, 1]

MODIFIED_BONDI = {1: 1.10, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 16: 1.80, 17: 1.75}


def _cavity_radius(atomic_number: int) -> float:
    return 1.2 * MODIFIED_BONDI[atomic_number]


def _default_module() -> DdcosmoReactionField:
    return DdcosmoReactionField(
        smearing_width=1.5, lebedev_order=29, max_spherical_harmonic_order=6
    )


def _pyscf_point_charge_ddcosmo_energy_ev(
    positions, charges, atomic_numbers, lebedev_order, lmax, eta, dielectric_constant
):
    """Reference ddCOSMO reaction-field energy (eV) for fixed point charges, via pyscf."""
    from pyscf import gto
    from pyscf.solvent import ddcosmo as pyscf_ddcosmo

    atoms = [(int(z), tuple(p)) for z, p in zip(atomic_numbers, positions, strict=False)]
    mol = gto.M(atom=atoms, unit="Angstrom", basis="sto-3g", verbose=0)
    solvent = pyscf_ddcosmo.DDCOSMO(mol)
    solvent.lebedev_order = lebedev_order
    solvent.lmax = lmax
    solvent.eta = eta
    solvent.eps = dielectric_constant
    radii_table = numpy.zeros(max(atomic_numbers) + 1)
    for atomic_number in set(atomic_numbers):
        radii_table[atomic_number] = _cavity_radius(atomic_number) / BOHR_IN_ANGSTROM  # -> Bohr
    solvent.radii_table = radii_table
    solvent.build()

    r_vdw = solvent._intermediates["r_vdw"]
    ylm_1sph = solvent._intermediates["ylm_1sph"]
    ui = solvent._intermediates["ui"]
    cosmo_matrix = solvent._intermediates["Lmat"]

    number_of_atoms = mol.natm
    number_of_lm = (lmax + 1) ** 2
    coordinates, weights = pyscf_ddcosmo.make_grids_one_sphere(lebedev_order)
    atom_coordinates = mol.atom_coords()  # Bohr
    charges = numpy.asarray(charges)

    right_hand_side = numpy.zeros((number_of_atoms, number_of_lm))
    for atom_index in range(number_of_atoms):
        cavity_points = atom_coordinates[atom_index] + r_vdw[atom_index] * coordinates
        distance = numpy.linalg.norm(
            cavity_points[:, None, :] - atom_coordinates[None, :, :], axis=2
        )
        potential = (charges[None, :] / distance).sum(axis=1)
        right_hand_side[atom_index] = -numpy.einsum(
            "n,xn,n->x", weights, ylm_1sph, ui[atom_index] * potential
        )
    surface_coefficients = numpy.linalg.solve(cosmo_matrix, right_hand_side.ravel()).reshape(
        number_of_atoms, number_of_lm
    )
    source_projection = numpy.zeros((number_of_atoms, number_of_lm))
    source_projection[:, 0] = numpy.sqrt(4 * numpy.pi) * charges / r_vdw
    f_epsilon = (dielectric_constant - 1.0) / dielectric_constant
    energy_hartree = 0.5 * f_epsilon * numpy.einsum(
        "jx,jx", source_projection, surface_coefficients
    )
    return energy_hartree * HARTREE_IN_ELECTRON_VOLT


def _tensors(positions, charges, atomic_numbers, batch):
    return (
        torch.tensor(charges),
        torch.tensor(positions),
        torch.tensor(atomic_numbers, dtype=torch.long),
        torch.tensor(batch, dtype=torch.long),
    )


def test_gas_limit_zero_energy_and_force():
    module = DdcosmoReactionField(smearing_width=1.5)
    charges, positions, atomic_numbers, batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    positions = positions.requires_grad_(True)
    gas_scaling = torch.tensor([dielectric_scaling(1.0)])
    energy = module(charges, positions, atomic_numbers, batch, gas_scaling)
    assert torch.equal(energy, torch.zeros_like(energy))
    (gradient,) = torch.autograd.grad(energy.sum(), positions)
    assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("lmax", [0, 2, 6])
@pytest.mark.parametrize("lebedev_order", [17, 29])
def test_matches_pyscf_point_charge_limit(lmax, lebedev_order):
    eta = 0.1
    dielectric_constant = 78.3553
    # tiny smearing width -> point-charge limit -> collapses onto pyscf's point-charge potential
    module = DdcosmoReactionField(
        smearing_width=1e-3,
        lebedev_order=lebedev_order,
        max_spherical_harmonic_order=lmax,
        regularization_eta=eta,
        solve_ridge=0.0,
    )
    charges, positions, atomic_numbers, batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    scaling = torch.tensor([dielectric_scaling(dielectric_constant)])
    module_energy = module(charges, positions, atomic_numbers, batch, scaling).item()
    reference_energy = _pyscf_point_charge_ddcosmo_energy_ev(
        WATER_POSITIONS,
        WATER_CHARGES,
        WATER_ATOMIC_NUMBERS,
        lebedev_order,
        lmax,
        eta,
        dielectric_constant,
    )
    assert module_energy < 0.0
    assert abs(module_energy - reference_energy) / abs(reference_energy) < 1e-6


def test_finite_difference_gradient():
    module = _default_module()
    charges, positions, atomic_numbers, batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    scaling = torch.tensor([dielectric_scaling(78.3553)])

    positions_grad = positions.clone().requires_grad_(True)
    energy = module(charges, positions_grad, atomic_numbers, batch, scaling).sum()
    (analytic_gradient,) = torch.autograd.grad(energy, positions_grad)

    step = 1e-5
    finite_difference = torch.zeros_like(positions)
    for atom_index in range(positions.shape[0]):
        for component in range(3):
            shifted_plus = positions.clone()
            shifted_plus[atom_index, component] += step
            shifted_minus = positions.clone()
            shifted_minus[atom_index, component] -= step
            energy_plus = module(charges, shifted_plus, atomic_numbers, batch, scaling).sum()
            energy_minus = module(charges, shifted_minus, atomic_numbers, batch, scaling).sum()
            finite_difference[atom_index, component] = (energy_plus - energy_minus) / (2 * step)

    assert torch.allclose(analytic_gradient, finite_difference, atol=1e-6)


def test_batch_isolation():
    module = _default_module()
    single_charges, single_positions, single_z, single_batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    scaling_a = dielectric_scaling(78.3553)
    scaling_b = dielectric_scaling(20.5)

    offset = torch.tensor([2.5, -1.5, 0.7])
    batched_positions = torch.cat([single_positions, single_positions + offset], dim=0)
    batched_charges = torch.cat([single_charges, single_charges])
    batched_z = torch.cat([single_z, single_z])
    batched_batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    batched_scaling = torch.tensor([scaling_a, scaling_b])

    batched_energy = module(
        batched_charges, batched_positions, batched_z, batched_batch, batched_scaling
    )
    energy_a = module(
        single_charges, single_positions, single_z, single_batch, torch.tensor([scaling_a])
    )
    energy_b = module(
        single_charges, single_positions, single_z, single_batch, torch.tensor([scaling_b])
    )
    assert torch.allclose(batched_energy[0], energy_a[0], atol=1e-10)
    assert torch.allclose(batched_energy[1], energy_b[0], atol=1e-10)


def test_eps_scaling_and_monotonicity():
    module = _default_module()
    charges, positions, atomic_numbers, batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    dielectric_ladder = [1.0, 2.02, 4.71, 20.5, 32.6, 78.3553]
    energy_over_f = []
    previous_magnitude = -1.0
    for dielectric_constant in dielectric_ladder:
        f_epsilon = dielectric_scaling(dielectric_constant)
        energy = module(
            charges, positions, atomic_numbers, batch, torch.tensor([f_epsilon])
        ).item()
        if dielectric_constant == 1.0:
            assert energy == 0.0
            continue
        assert energy < 0.0
        assert abs(energy) > previous_magnitude  # monotonic strengthening with eps
        previous_magnitude = abs(energy)
        energy_over_f.append(energy / f_epsilon)
    assert max(energy_over_f) - min(energy_over_f) < 1e-9


def test_torchscript_matches_eager():
    module = _default_module()
    scripted = torch.jit.script(module)
    charges, positions, atomic_numbers, batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    scaling = torch.tensor([dielectric_scaling(78.3553)])
    eager_energy = module(charges, positions, atomic_numbers, batch, scaling)
    scripted_energy = scripted(charges, positions, atomic_numbers, batch, scaling)
    assert torch.allclose(eager_energy, scripted_energy, atol=1e-10)
