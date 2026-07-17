"""Tests for the screened generalized-Born (GB) monopole reaction-field scaffold.

Independent-solver control: OpenMM is NOT installed in this environment (checked:
``import openmm`` fails), so the independent reference is a self-contained pure-numpy
implementation of the same physics -- OBC-II effective Born radii from the
Hawkins-Cramer-Truhlar pairwise descreening integral plus the (erf-screened) Still
generalized-Born energy. It shares no code with the torch module, so agreement validates the
torch implementation. A second check confirms the classic (unscreened) Still point-charge GB
is recovered as the smearing width -> 0.
"""

from __future__ import annotations

import math

import numpy
import pytest
import torch
from mace_scf.electrostatics.reaction_field import (
    COULOMB_CONSTANT_EV_ANGSTROM,
    ScreenedGeneralizedBornSolvation,
    dielectric_scaling,
)
from mace_scf.electrostatics.reaction_field.screened_potential import build_intramolecular_pairs


@pytest.fixture(autouse=True)
def _use_double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


# --- fixed test molecule (water-like, atom order O, H, H) ------------------------------------
WATER_POSITIONS = [
    [0.0, 0.0, 0.117],
    [0.0, 0.757, -0.469],
    [0.0, -0.757, -0.469],
]
WATER_CHARGES = [-0.8, 0.4, 0.4]
WATER_ATOMIC_NUMBERS = [8, 1, 1]

MODIFIED_BONDI = {1: 1.10, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 16: 1.80, 17: 1.75}

# module construction defaults, mirrored by the numpy reference
BORN_SCALE = 0.8
OBC_ALPHA = 1.0
OBC_BETA = 0.8
OBC_GAMMA = 4.85
OFFSET = 0.09
BORN_MIN = 0.1
BORN_MAX = 30.0


def _cavity_radius(atomic_number: int) -> float:
    return 1.2 * MODIFIED_BONDI[atomic_number]


def _reference_born_radii(positions, atomic_numbers):
    positions = numpy.asarray(positions)
    intrinsic = numpy.asarray([_cavity_radius(z) for z in atomic_numbers])
    offset_radius = intrinsic - OFFSET
    number_of_atoms = len(atomic_numbers)
    integral = numpy.zeros(number_of_atoms)
    for i in range(number_of_atoms):
        for j in range(number_of_atoms):
            if i == j:
                continue
            distance = numpy.linalg.norm(positions[i] - positions[j])
            scaled_radius = BORN_SCALE * offset_radius[j]
            upper = distance + scaled_radius
            lower = max(offset_radius[i], abs(distance - scaled_radius))
            if upper > offset_radius[i]:
                integral[i] += 0.5 * (
                    1.0 / lower
                    - 1.0 / upper
                    + 0.25
                    * (distance - scaled_radius * scaled_radius / distance)
                    * (1.0 / upper**2 - 1.0 / lower**2)
                    + 0.5 * math.log(lower / upper) / distance
                )
    psi = integral * offset_radius
    tanh_argument = OBC_ALPHA * psi - OBC_BETA * psi**2 + OBC_GAMMA * psi**3
    born = 1.0 / (1.0 / offset_radius - numpy.tanh(tanh_argument) / intrinsic)
    return numpy.clip(born, BORN_MIN, BORN_MAX)


def _reference_gb_energy(positions, charges, atomic_numbers, smearing_width, f_epsilon):
    positions = numpy.asarray(positions)
    charges = numpy.asarray(charges)
    born = _reference_born_radii(positions, atomic_numbers)
    energy = 0.0
    for i in range(len(charges)):
        for j in range(len(charges)):
            distance = numpy.linalg.norm(positions[i] - positions[j])
            radius_product = born[i] * born[j]
            effective_distance = math.sqrt(
                distance**2 + radius_product * math.exp(-(distance**2) / (4.0 * radius_product))
            )
            if smearing_width is None:
                kernel = 1.0 / effective_distance  # classic unscreened Still GB
            else:
                kernel = math.erf(0.5 * effective_distance / smearing_width) / effective_distance
            energy += (
                -0.5
                * COULOMB_CONSTANT_EV_ANGSTROM
                * f_epsilon
                * charges[i]
                * charges[j]
                * kernel
            )
    return energy


def _tensors(positions, charges, atomic_numbers, batch):
    return (
        torch.tensor(charges),
        torch.tensor(positions),
        torch.tensor(atomic_numbers, dtype=torch.long),
        torch.tensor(batch, dtype=torch.long),
    )


def test_gas_limit_zero_energy_and_force():
    module = ScreenedGeneralizedBornSolvation(smearing_width=1.5)
    charges, positions, atomic_numbers, batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    positions = positions.requires_grad_(True)
    gas_scaling = torch.tensor([dielectric_scaling(1.0)])
    energy = module(charges, positions, atomic_numbers, batch, gas_scaling)
    assert torch.equal(energy, torch.zeros_like(energy))
    (gradient,) = torch.autograd.grad(energy.sum(), positions)
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_matches_independent_numpy_reference():
    smearing_width = 1.5
    module = ScreenedGeneralizedBornSolvation(smearing_width=smearing_width)
    charges, positions, atomic_numbers, batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    f_epsilon = dielectric_scaling(78.3553)
    module_energy = module(
        charges, positions, atomic_numbers, batch, torch.tensor([f_epsilon])
    ).item()

    # Born radii and total energy both match the independent numpy solver.
    reference_born = _reference_born_radii(WATER_POSITIONS, WATER_ATOMIC_NUMBERS)
    module_born = module.compute_born_radii(positions, atomic_numbers, batch).detach().numpy()
    assert numpy.allclose(module_born, reference_born, rtol=1e-10, atol=1e-10)

    reference_energy = _reference_gb_energy(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, smearing_width, f_epsilon
    )
    assert module_energy < 0.0  # stabilizing
    assert abs(module_energy - reference_energy) < 1e-9


def test_small_width_recovers_unscreened_still_gb():
    # As smearing width -> 0 the erf screening -> 1: classic point-charge Still GB.
    smearing_width = 1e-3
    module = ScreenedGeneralizedBornSolvation(smearing_width=smearing_width)
    charges, positions, atomic_numbers, batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    f_epsilon = dielectric_scaling(78.3553)
    module_energy = module(
        charges, positions, atomic_numbers, batch, torch.tensor([f_epsilon])
    ).item()
    classic_energy = _reference_gb_energy(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, None, f_epsilon
    )
    assert abs(module_energy - classic_energy) < 1e-6


def test_finite_difference_gradient():
    module = ScreenedGeneralizedBornSolvation(smearing_width=1.5)
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

    assert torch.allclose(analytic_gradient, finite_difference, atol=1e-7)


def test_batch_isolation():
    module = ScreenedGeneralizedBornSolvation(smearing_width=1.5)
    single_charges, single_positions, single_z, single_batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    scaling_a = dielectric_scaling(78.3553)
    scaling_b = dielectric_scaling(20.5)

    # second molecule placed far away and, deliberately, also close-by shifted; isolation must
    # hold regardless of where molecule B sits in the shared coordinate space.
    offset = torch.tensor([3.0, -2.0, 1.0])
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
    assert torch.allclose(batched_energy[0], energy_a[0], atol=1e-12)
    assert torch.allclose(batched_energy[1], energy_b[0], atol=1e-12)


def test_eps_scaling_follows_f_epsilon():
    module = ScreenedGeneralizedBornSolvation(smearing_width=1.5)
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
        # monotonic strengthening with eps
        assert abs(energy) > previous_magnitude
        previous_magnitude = abs(energy)
        energy_over_f.append(energy / f_epsilon)
    # energy scales exactly like f(eps): E/f is constant.
    assert max(energy_over_f) - min(energy_over_f) < 1e-9


def test_torchscript_matches_eager():
    module = ScreenedGeneralizedBornSolvation(smearing_width=1.5)
    scripted = torch.jit.script(module)
    charges, positions, atomic_numbers, batch = _tensors(
        WATER_POSITIONS, WATER_CHARGES, WATER_ATOMIC_NUMBERS, [0, 0, 0]
    )
    scaling = torch.tensor([dielectric_scaling(78.3553)])
    eager_energy = module(charges, positions, atomic_numbers, batch, scaling)
    scripted_energy = scripted(charges, positions, atomic_numbers, batch, scaling)
    assert torch.allclose(eager_energy, scripted_energy, atol=1e-12)


def test_intramolecular_pairs_match_graph_longrange():
    # the scriptable pair builder reproduces graph_longrange's pair set (block-diagonal).
    from graph_longrange.realspace_electrostatics import (
        batch_complete_graph_excluding_self_duplicates_vector,
    )

    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    ours = build_intramolecular_pairs(batch)
    reference = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)
    ours_set = {(int(s), int(r)) for s, r in zip(ours[0], ours[1], strict=False)}
    reference_set = {
        (int(s), int(r)) for s, r in zip(reference[0], reference[1], strict=False)
    }
    assert ours_set == reference_set
