"""Screened generalized-Born (GB) monopole reaction field -- the scaffold / Exp-4 baseline.

This is NOT the production model (ddCOSMO is). It is a permanent sanity baseline that
exercises the same interface: a one-shot, autograd-differentiable, batch-isolated,
TorchScript-scriptable functional of predicted atomic monopole charges + geometry + dielectric
scaling.

Physics
-------
* OBC-style effective Born radii (Onufriev-Bashford-Case, OBC-II parameters) built from the
  Hawkins-Cramer-Truhlar pairwise descreening integral, exactly the recipe used by OpenMM's
  ``GBSAOBCForce`` / the ``CustomGBForce`` OBC example. The Born radii are CLAMPED away from
  0 / negative / huge values (they are numerically ill-behaved by construction).
* Still's generalized-Born pair function ``f_GB`` with an added erf sigma-screening to match
  the Gaussian-smeared MACE density (single source of the screened kernel:
  :mod:`reaction_field.screened_potential`). As the smearing width -> 0 the screening -> 1 and
  the classic (unscreened) Still point-charge GB is recovered.
* Energy ``dG_GB = -0.5 * f(eps) * k * sum_{i,j} q_i q_j * erf(0.5 f_GB / width) / f_GB``,
  summed over all ordered same-molecule pairs (i == j is the Born self term). ``k`` is the
  Coulomb constant in eV*Angstrom/e**2, ``f(eps) = (eps-1)/eps`` the dielectric scaling. The
  intrinsic Born radii use the SAME cavity radius table as ddCOSMO
  (``1.2 * modified_Bondi[Z]``) so the two schemes share one cavity source.

Units: eV / Angstrom / elementary-charge. ``dG_GB < 0`` for a polar solute in polar solvent.
"""

from __future__ import annotations

from typing import Optional

import torch
from mace.tools.scatter import scatter_sum
from torch import Tensor

from .screened_potential import (
    COULOMB_CONSTANT_EV_ANGSTROM,
    build_cavity_radius_lookup,
    build_intramolecular_pairs,
    generalized_born_effective_distance,
    screened_coulomb_kernel,
)


class ScreenedGeneralizedBornSolvation(torch.nn.Module):
    """Screened generalized-Born monopole reaction-field energy (per graph).

    forward(charges, positions, atomic_numbers, batch, dielectric_scaling) -> ``[n_graphs]``

    Args of ``forward``:
        charges: ``[n_atoms]`` predicted atomic monopole charges (elementary charge). For the
            MACE density these are the physical l=0 charges (no e3nn rescale needed at l=0).
        positions: ``[n_atoms, 3]`` Angstrom.
        atomic_numbers: ``[n_atoms]`` long; used to look up the intrinsic cavity/Born radius.
        batch: ``[n_atoms]`` long; graph id of each atom (block-diagonal isolation).
        dielectric_scaling: ``[n_graphs]`` the per-graph ``f(eps) = (eps-1)/eps``. At ``eps==1``
            pass 0.0 -> the returned energy is exactly 0 (and so is its gradient).

    Returns:
        ``[n_graphs]`` reaction-field free energy in eV.

    Construction:
        smearing_width: Gaussian width of the solute density (Angstrom); the same value the
            vacuum electrostatics uses (design default sigma = 1.5 Angstrom).
        born_scale: uniform HCT descreening scale factor S applied to every atom (OpenMM/Amber
            use per-element factors; a single documented value is enough for a scaffold).
        obc_alpha / obc_beta / obc_gamma: OBC tanh parameters (defaults are OBC-II: 1.0, 0.8,
            4.85).
        offset_radius_angstrom: HCT dielectric offset (0.09 Angstrom == OpenMM's 0.009 nm).
        born_radius_min_angstrom / born_radius_max_angstrom: clamp bounds for the effective
            Born radii.
    """

    def __init__(
        self,
        smearing_width: float,
        born_scale: float = 0.8,
        obc_alpha: float = 1.0,
        obc_beta: float = 0.8,
        obc_gamma: float = 4.85,
        offset_radius_angstrom: float = 0.09,
        born_radius_min_angstrom: float = 0.1,
        born_radius_max_angstrom: float = 30.0,
    ) -> None:
        super().__init__()
        self.smearing_width = float(smearing_width)
        self.born_scale = float(born_scale)
        self.obc_alpha = float(obc_alpha)
        self.obc_beta = float(obc_beta)
        self.obc_gamma = float(obc_gamma)
        self.offset_radius_angstrom = float(offset_radius_angstrom)
        self.born_radius_min_angstrom = float(born_radius_min_angstrom)
        self.born_radius_max_angstrom = float(born_radius_max_angstrom)
        self.coulomb_constant = COULOMB_CONSTANT_EV_ANGSTROM
        # Intrinsic (pre-offset) cavity radius per atomic number; NaN for unsupported Z.
        self.register_buffer("cavity_radius_lookup", build_cavity_radius_lookup())

    def compute_born_radii(
        self,
        positions: Tensor,
        atomic_numbers: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """OBC-II effective Born radii per atom, ``[n_atoms]`` (Angstrom), block-isolated.

        Uses the HCT pairwise descreening integral over same-molecule neighbours and the OBC
        tanh rescaling, then clamps to ``[born_radius_min, born_radius_max]``.
        """
        intrinsic_radius = self.cavity_radius_lookup.index_select(0, atomic_numbers)
        offset_radius = intrinsic_radius - self.offset_radius_angstrom

        edge_index = build_intramolecular_pairs(batch)
        sender = edge_index[0]
        receiver = edge_index[1]

        number_of_atoms = positions.shape[0]
        if sender.numel() == 0:
            # No neighbours (isolated atoms): integral is zero, Born radius == offset radius.
            descreening_integral = torch.zeros(
                number_of_atoms, dtype=positions.dtype, device=positions.device
            )
        else:
            displacement = positions.index_select(0, receiver) - positions.index_select(0, sender)
            distance = torch.linalg.norm(displacement, dim=1)  # [n_edges]
            offset_radius_receiver = offset_radius.index_select(0, receiver)
            scaled_radius_sender = self.born_scale * offset_radius.index_select(0, sender)

            upper = distance + scaled_radius_sender
            difference = torch.abs(distance - scaled_radius_sender)
            lower = torch.maximum(offset_radius_receiver, difference)
            # Only contributes when the descreening sphere reaches into atom i's region.
            gate = (upper > offset_radius_receiver).to(positions.dtype)

            inverse_lower = 1.0 / lower
            inverse_upper = 1.0 / upper
            edge_term = gate * 0.5 * (
                inverse_lower
                - inverse_upper
                + 0.25
                * (distance - scaled_radius_sender * scaled_radius_sender / distance)
                * (inverse_upper * inverse_upper - inverse_lower * inverse_lower)
                + 0.5 * torch.log(lower / upper) / distance
            )
            descreening_integral = scatter_sum(
                src=edge_term, index=receiver, dim=0, dim_size=number_of_atoms
            )

        psi = descreening_integral * offset_radius
        tanh_argument = (
            self.obc_alpha * psi
            - self.obc_beta * psi * psi
            + self.obc_gamma * psi * psi * psi
        )
        inverse_born_radius = 1.0 / offset_radius - torch.tanh(tanh_argument) / intrinsic_radius
        # inverse_born_radius can go <= 0 (huge / negative Born radii): clamp on the radius.
        born_radius = 1.0 / inverse_born_radius
        born_radius = torch.clamp(
            born_radius,
            min=self.born_radius_min_angstrom,
            max=self.born_radius_max_angstrom,
        )
        return born_radius

    def forward(
        self,
        charges: Tensor,
        positions: Tensor,
        atomic_numbers: Tensor,
        batch: Tensor,
        dielectric_scaling: Tensor,
        atomic_dipoles: Optional[Tensor] = None,
    ) -> Tensor:
        # ``atomic_dipoles`` is accepted for a uniform reaction-field interface but ignored:
        # generalized-Born is intrinsically a monopole (Born) model. The disposable GB baseline
        # stays monopole-only by construction.
        number_of_graphs = int(dielectric_scaling.shape[0])
        born_radius = self.compute_born_radii(positions, atomic_numbers, batch)

        # --- Born self term (i == j): -0.5 * k * q_i^2 * erf(0.5 R_i / width) / R_i ---
        self_kernel = screened_coulomb_kernel(born_radius, self.smearing_width)
        self_energy_per_atom = -0.5 * self.coulomb_constant * charges * charges * self_kernel
        energy = scatter_sum(
            src=self_energy_per_atom, index=batch, dim=0, dim_size=number_of_graphs
        )

        # --- pair term (i != j, same molecule) over the complete block-diagonal graph ---
        edge_index = build_intramolecular_pairs(batch)
        sender = edge_index[0]
        receiver = edge_index[1]
        if sender.numel() > 0:
            displacement = positions.index_select(0, receiver) - positions.index_select(0, sender)
            distance = torch.linalg.norm(displacement, dim=1)
            effective_distance = generalized_born_effective_distance(
                distance,
                born_radius.index_select(0, sender),
                born_radius.index_select(0, receiver),
            )
            pair_kernel = screened_coulomb_kernel(effective_distance, self.smearing_width)
            # -0.5 * k * q_i q_j * kernel; each unordered pair appears twice (i->j and j->i)
            # so the 0.5 combines to the full pair energy.
            edge_energy = (
                -0.5
                * self.coulomb_constant
                * charges.index_select(0, sender)
                * charges.index_select(0, receiver)
                * pair_kernel
            )
            pair_energy = scatter_sum(
                src=edge_energy, index=receiver, dim=0, dim_size=positions.shape[0]
            )
            energy = energy + scatter_sum(
                src=pair_energy, index=batch, dim=0, dim_size=number_of_graphs
            )

        return energy * dielectric_scaling
