"""Shared screened-Coulomb / screened-Gaussian helpers for the reaction-field modules.

This is the SINGLE source of the erf-screened Coulomb kernel on the reaction-field side. It
reuses the exact convention of the vacuum electrostatics in ``graph_longrange`` so the solute
potential seen by the continuum surface is consistent with the density the model already
predicts:

* ``graph_longrange.realspace_electrostatics.charges_energy_from_graph`` screens the pair
  interaction of two Gaussians of width ``density_smearing_width`` as
  ``erf(distance * 0.5 / density_smearing_width) / distance`` and scales the energy by
  ``FIELD_CONSTANT / (4 * pi)``.
* ``graph_longrange.realspace_electrostatics.charges_features_from_graph`` evaluates the
  potential of such Gaussians at (point) receivers with the combined width factor
  ``sqrt((density_smearing_width**2 + projection_smearing_width**2) / 2)``; for a point
  receiver (``projection_smearing_width == 0``) that is ``density_smearing_width / sqrt(2)``,
  i.e. the erf argument becomes ``distance / (sqrt(2) * density_smearing_width)``.

Everything is in eV / Angstrom / elementary-charge. The Coulomb constant is
``FIELD_CONSTANT / (4 * pi) ~= 14.3996`` eV*Angstrom/e**2.

All functions are TorchScript-scriptable (plain tensor ops, explicit type annotations).
"""

from __future__ import annotations

import math

import torch

# Reuse the vacuum-electrostatics Coulomb constant rather than defining a second one.
from graph_longrange.utils import FIELD_CONSTANT
from torch import Tensor

from .constants import (
    MODIFIED_BONDI_RADII_ANGSTROM,
    cavity_radius_angstrom,
    highest_supported_atomic_number,
)

# Coulomb constant in eV * Angstrom / e**2 (~= 14.3996). Same value used by graph_longrange
# (there it appears as FIELD_CONSTANT / (4 * pi) inside the energy/feature routines).
COULOMB_CONSTANT_EV_ANGSTROM: float = FIELD_CONSTANT / (4.0 * math.pi)

# Distance floor guarding 1/r and erf(x)/x at r -> 0. The erf numerator already vanishes
# linearly at r -> 0, so the kernel stays finite; the floor only removes the exact 0/0.
_DISTANCE_FLOOR: float = 1.0e-12


def build_cavity_radius_lookup(dtype: torch.dtype | None = None) -> Tensor:
    """Build a per-atomic-number cavity-radius lookup tensor for use as a registered buffer.

    Returns a 1-D tensor indexed by atomic number (``lookup[Z]``), size
    ``highest_supported_atomic_number() + 1``, holding ``cavity_radius_angstrom(Z)`` for the
    frozen elements (H, C, N, O, F, S, Cl) and ``NaN`` elsewhere. Indexing an unsupported Z
    therefore yields ``NaN`` which propagates loudly instead of silently mismatching the
    reference cavity. (A scriptable ``forward`` cannot ``raise`` on bad Z; the NaN is the guard.)

    ``dtype`` defaults to the current default dtype resolved at CALL time (not import time), so a
    module constructed under ``torch.set_default_dtype(torch.float64)`` gets float64 radii.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    size = highest_supported_atomic_number() + 1
    lookup = torch.full((size,), float("nan"), dtype=dtype)
    for atomic_number in MODIFIED_BONDI_RADII_ANGSTROM:
        lookup[atomic_number] = cavity_radius_angstrom(atomic_number)
    return lookup


def build_intramolecular_pairs(batch: Tensor) -> Tensor:
    """Directed same-molecule atom pairs ``(sender, receiver)`` with ``sender != receiver``.

    Returns ``[2, n_pairs]`` (row 0 = sender, row 1 = receiver). Equivalent to
    ``graph_longrange.realspace_electrostatics.
    batch_complete_graph_excluding_self_duplicates_vector(batch, 1)`` (verified to yield the same
    pair set), but reimplemented here because that function's ``@torch.no_grad()`` decorator
    prevents TorchScript compilation. Cross-molecule pairs are excluded, so interactions stay
    block-diagonal per molecule.
    """
    number_of_atoms = batch.shape[0]
    atom_index = torch.arange(number_of_atoms, device=batch.device)
    sender = atom_index.view(-1, 1).expand(number_of_atoms, number_of_atoms).reshape(-1)
    receiver = atom_index.view(1, -1).expand(number_of_atoms, number_of_atoms).reshape(-1)
    same_molecule = (batch.view(-1, 1) == batch.view(1, -1)).reshape(-1)
    keep = same_molecule & (sender != receiver)
    return torch.stack(
        [torch.masked_select(sender, keep), torch.masked_select(receiver, keep)], dim=0
    )


def dielectric_scaling_from_dielectric_constant(dielectric_constant: Tensor) -> Tensor:
    """Tensor-valued C-PCM / ddCOSMO dielectric scaling ``f(eps) = (eps - 1) / eps``.

    Args:
        dielectric_constant: per-graph dielectric constant ``eps``, shape ``[n_graphs]``.

    Returns:
        ``f(eps)`` per graph, shape ``[n_graphs]``. Exactly 0.0 where ``eps == 1`` (gas limit,
        no reaction field). ``eps -> inf`` gives 1.0 (conductor). ``eps`` is assumed finite and
        ``>= 1`` (the reference ladder never passes non-physical values).

    The model forward methods take ``dielectric_scaling`` (already ``f(eps)``) directly; this
    helper is exposed so callers that hold raw ``eps`` can convert once.
    """
    return (dielectric_constant - 1.0) / dielectric_constant


def screened_coulomb_kernel(distance: Tensor, effective_smearing_width: float) -> Tensor:
    """erf-screened reciprocal distance ``erf(0.5 * distance / width) / distance``.

    This is the ``graph_longrange`` convention. Choose ``effective_smearing_width`` per use:

    * two Gaussians each of width ``w`` (source-source, e.g. the GB pair/self term):
      ``effective_smearing_width = w`` -> ``erf(0.5 * distance / w) / distance``;
    * a Gaussian of width ``w`` evaluated at a bare point (source-point, e.g. the solute
      potential on a cavity grid point): ``effective_smearing_width = w / sqrt(2)`` ->
      ``erf(distance / (sqrt(2) * w)) / distance``.

    The kernel is finite at ``distance -> 0`` (the erf numerator vanishes linearly); the
    distance floor (:data:`_DISTANCE_FLOOR`, inlined here so the function stays scriptable)
    only removes the exact 0/0 so autograd stays clean.
    """
    guarded_distance = distance + 1.0e-12
    return torch.erf(0.5 * distance / effective_smearing_width) / guarded_distance


def point_target_effective_width(smearing_width: float) -> float:
    """Effective width for evaluating a width-``smearing_width`` Gaussian at a bare point."""
    return smearing_width / math.sqrt(2.0)


def screened_gaussian_potential_at_points(
    target_points: Tensor,
    target_batch: Tensor,
    source_charges: Tensor,
    source_positions: Tensor,
    source_batch: Tensor,
    smearing_width: float,
    coulomb_constant: float = COULOMB_CONSTANT_EV_ANGSTROM,
) -> Tensor:
    """Screened-Gaussian solute potential evaluated at arbitrary target points.

    Each source is a Gaussian charge of width ``smearing_width``; each target is a bare point.
    Interactions are confined WITHIN a molecule via the batch indices (a target only sees
    sources sharing its ``batch`` value), so molecules in a batch never leak into each other.

    Args:
        target_points: ``[n_targets, 3]`` Angstrom.
        target_batch: ``[n_targets]`` long, graph id of each target.
        source_charges: ``[n_sources]`` elementary charge.
        source_positions: ``[n_sources, 3]`` Angstrom.
        source_batch: ``[n_sources]`` long, graph id of each source.
        smearing_width: Gaussian width of the source density (Angstrom).
        coulomb_constant: Coulomb constant in eV*Angstrom/e**2 (defaults to the reused
            graph_longrange value; a default parameter rather than a module-global reference so
            the function stays TorchScript-scriptable).

    Returns:
        ``[n_targets]`` potential in eV/e (already includes the Coulomb constant).
    """
    displacement = target_points.unsqueeze(1) - source_positions.unsqueeze(0)  # [T, S, 3]
    distance = torch.linalg.norm(displacement, dim=2)  # [T, S]
    kernel = screened_coulomb_kernel(distance, point_target_effective_width(smearing_width))
    same_molecule = target_batch.unsqueeze(1) == source_batch.unsqueeze(0)  # [T, S]
    kernel = torch.where(same_molecule, kernel, torch.zeros_like(kernel))
    potential = (source_charges.unsqueeze(0) * kernel).sum(dim=1)  # [T]
    return coulomb_constant * potential


def generalized_born_effective_distance(
    interatomic_distance: Tensor,
    born_radius_sender: Tensor,
    born_radius_receiver: Tensor,
) -> Tensor:
    """Still's generalized-Born effective distance ``f_GB``.

    ``f_GB = sqrt(r**2 + R_i * R_j * exp(-r**2 / (4 * R_i * R_j)))``. For ``r = 0`` (the Born
    self term) this collapses to ``sqrt(R_i * R_j)``; with equal radii that is the Born radius,
    giving the usual self-energy ``-0.5 * f(eps) * k * q**2 / R``.
    """
    radius_product = born_radius_sender * born_radius_receiver
    squared_distance = interatomic_distance * interatomic_distance
    return torch.sqrt(
        squared_distance + radius_product * torch.exp(-squared_distance / (4.0 * radius_product))
    )
