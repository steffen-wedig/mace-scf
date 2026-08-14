"""Generalized-Kirkwood (GK) reaction-field provider for a charge + atomic-dipole solute.

The multipole extension of generalized Born: where GB couples only to atomic charges, GK couples
to charges **and** atomic dipoles, with the order-dependent Kirkwood dielectric factors

    f_0 = (eps - 1) / eps               (Born, monopole)
    f_1 = 2 (eps - 1) / (2 eps + 1)     (Onsager, dipole)

so the dipole channel gets its physically correct, distinct scaling -- this is what lets the
solvent reaction *field* drive dipole enhancement, which the monopole-only GB cannot. The tiers
share the OBC-II effective radii and the block-diagonal intramolecular pair graph; GK swaps the
Still-GB Gaussian constant ``c = 4`` for the AMOEBA value ``gkc = 2.455`` (Corrigan et al. 2023).

Closed forms (Schnieders & Ponder 2007; Corrigan et al. 2023, verified against the Tinker ``egk``
reference), per atom ``i`` with ``r_ij = pos_j - pos_i``, effective distance ``f = f_GK`` and
geometric descreening factor ``expc1``:

    phi_i = eps0_i q_i / a_i
          + sum_{j != i} [ eps0_i q_j / f  -  eps01 (mu_j . r_ij) / f**3 ]

    E_i   = eps1_i mu_i / a_i**3
          + sum_{j != i} [ eps01 r_ij q_j / f**3
                         + eps1_i ( mu_j / f**3  -  3 expc1 r_ij (mu_j . r_ij) / f**5 ) ]

with ``eps0_i = -k f_0``, ``eps1_i = -k f_1`` (``k`` the Coulomb constant, sign making G < 0), and
the **symmetrised** charge-dipole factor ``eps01 = 1/2 (eps0 expc1 + eps1)`` -- the Tinker-verified
resolution of the charge-dipole non-reciprocity (a single order factor would give a non-symmetric
interaction and inconsistent forces). The self terms reproduce the analytic Born
(``-1/2 k f_0 q**2 / a``) and Onsager (``-1/2 k f_1 |mu|**2 / a**3``) reaction energies.

The charge-dipole and dipole-dipole tensors form a **symmetric** interaction ``K`` (so
``phi = dG/dq`` and ``E = dG/dmu`` hold; enforced by test). NOTE: these are the *point-multipole*
Kirkwood tensors (kernel ``1/f``); the erf-screening of the l=1 tensors to match the Gaussian-
smeared MACE density (width w) is a documented refinement, not yet applied here -- the effective
radii keep ``f`` bounded away from zero so the point form is numerically well behaved.

Everything is eV / Angstrom / elementary-charge; ``eps == 1`` (gas) zeroes every factor. Pure
tensor ops, TorchScript-scriptable.
"""

from __future__ import annotations

import torch
from mace.tools.scatter import scatter_sum
from torch import Tensor

from .reaction_potential import ReactionFieldProvider, reaction_free_energy
from .screened_potential import (
    COULOMB_CONSTANT_EV_ANGSTROM,
    build_cavity_radius_lookup,
    build_intramolecular_pairs,
    compute_obc_born_radii,
    generalized_born_descreening_factor,
    generalized_born_effective_distance,
    kirkwood_dielectric_factor,
)

# AMOEBA generalized-Kirkwood Gaussian constant (Corrigan et al. 2023); Still's GB uses 4.0.
GENERALIZED_KIRKWOOD_GAUSSIAN_CONSTANT: float = 2.455


def generalized_kirkwood_potential_and_field(
    monopole_charges: Tensor,
    atomic_dipoles: Tensor,
    positions: Tensor,
    batch: Tensor,
    born_radius: Tensor,
    monopole_prefactor: Tensor,
    dipole_prefactor: Tensor,
    gaussian_constant: float,
) -> tuple[Tensor, Tensor]:
    """Per-atom GK reaction potential ``phi[n_atoms]`` and field ``E[n_atoms, 3]``.

    ``monopole_prefactor`` and ``dipole_prefactor`` are the per-atom ``eps0 = -k f_0`` and
    ``eps1 = -k f_1`` (Coulomb constant and sign folded in). ``born_radius`` is the per-atom
    effective radius. Split out from the provider so tests can drive it with explicit radii.
    """
    number_of_atoms = positions.shape[0]

    # --- self terms (Born monopole + Onsager dipole) ---
    reaction_potential = monopole_prefactor * monopole_charges / born_radius
    reaction_field = dipole_prefactor.unsqueeze(-1) * atomic_dipoles / (
        born_radius * born_radius * born_radius
    ).unsqueeze(-1)

    # --- pair terms over the block-diagonal intramolecular graph ---
    edge_index = build_intramolecular_pairs(batch)
    sender = edge_index[0]
    receiver = edge_index[1]
    if sender.numel() > 0:
        # r_ij with i = receiver, j = sender.
        r_ij = positions.index_select(0, sender) - positions.index_select(0, receiver)
        distance = torch.linalg.norm(r_ij, dim=1)
        radius_sender = born_radius.index_select(0, sender)
        radius_receiver = born_radius.index_select(0, receiver)
        effective_distance = generalized_born_effective_distance(
            distance, radius_sender, radius_receiver, gaussian_constant
        )
        descreening = generalized_born_descreening_factor(
            distance, radius_sender, radius_receiver, gaussian_constant
        )

        inverse_f = 1.0 / effective_distance
        inverse_f3 = inverse_f * inverse_f * inverse_f
        inverse_f5 = inverse_f3 * inverse_f * inverse_f

        eps0 = monopole_prefactor.index_select(0, receiver)
        eps1 = dipole_prefactor.index_select(0, receiver)
        eps01 = 0.5 * (eps0 * descreening + eps1)

        charge_sender = monopole_charges.index_select(0, sender)
        dipole_sender = atomic_dipoles.index_select(0, sender)
        dipole_dot_r = (dipole_sender * r_ij).sum(dim=-1)  # [n_pairs]

        # potential at receiver from sender's charge and dipole
        potential_pair = eps0 * charge_sender * inverse_f - eps01 * dipole_dot_r * inverse_f3
        reaction_potential = reaction_potential + scatter_sum(
            src=potential_pair, index=receiver, dim=0, dim_size=number_of_atoms
        )

        # field at receiver from sender's charge and dipole
        field_from_charge = (eps01 * charge_sender * inverse_f3).unsqueeze(-1) * r_ij
        field_from_dipole = eps1.unsqueeze(-1) * (
            dipole_sender * inverse_f3.unsqueeze(-1)
            - 3.0 * descreening.unsqueeze(-1) * r_ij * (dipole_dot_r * inverse_f5).unsqueeze(-1)
        )
        field_pair = field_from_charge + field_from_dipole
        reaction_field = reaction_field + scatter_sum(
            src=field_pair, index=receiver, dim=0, dim_size=number_of_atoms
        )

    return reaction_potential, reaction_field


class GeneralizedKirkwoodReactionField(ReactionFieldProvider):
    """Generalized-Kirkwood provider: charge + atomic-dipole reaction potential and field."""

    def __init__(
        self,
        smearing_width: float,
        gaussian_constant: float = GENERALIZED_KIRKWOOD_GAUSSIAN_CONSTANT,
        born_scale: float = 0.8,
        obc_alpha: float = 1.0,
        obc_beta: float = 0.8,
        obc_gamma: float = 4.85,
        offset_radius_angstrom: float = 0.09,
        born_radius_min_angstrom: float = 0.1,
        born_radius_max_angstrom: float = 30.0,
    ) -> None:
        super().__init__()
        # ``smearing_width`` is accepted for interface parity with the screened tiers and for the
        # future screened-l=1 refinement; the point-multipole tensors here do not yet use it.
        self.smearing_width = float(smearing_width)
        self.gaussian_constant = float(gaussian_constant)
        self.born_scale = float(born_scale)
        self.obc_alpha = float(obc_alpha)
        self.obc_beta = float(obc_beta)
        self.obc_gamma = float(obc_gamma)
        self.offset_radius_angstrom = float(offset_radius_angstrom)
        self.born_radius_min_angstrom = float(born_radius_min_angstrom)
        self.born_radius_max_angstrom = float(born_radius_max_angstrom)
        self.coulomb_constant = COULOMB_CONSTANT_EV_ANGSTROM
        self.register_buffer("cavity_radius_lookup", build_cavity_radius_lookup())

    def forward(
        self,
        monopole_charges: Tensor,
        atomic_dipoles: Tensor,
        positions: Tensor,
        atomic_numbers: Tensor,
        batch: Tensor,
        dielectric_constant: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        number_of_graphs = int(dielectric_constant.shape[0])
        born_radius = compute_obc_born_radii(
            positions,
            atomic_numbers,
            batch,
            self.cavity_radius_lookup,  # ty: ignore[invalid-argument-type]
            self.born_scale,
            self.obc_alpha,
            self.obc_beta,
            self.obc_gamma,
            self.offset_radius_angstrom,
            self.born_radius_min_angstrom,
            self.born_radius_max_angstrom,
        )
        monopole_scaling = kirkwood_dielectric_factor(dielectric_constant, 0)
        dipole_scaling = kirkwood_dielectric_factor(dielectric_constant, 1)
        monopole_prefactor = -self.coulomb_constant * monopole_scaling.index_select(0, batch)
        dipole_prefactor = -self.coulomb_constant * dipole_scaling.index_select(0, batch)

        reaction_potential, reaction_field = generalized_kirkwood_potential_and_field(
            monopole_charges,
            atomic_dipoles,
            positions,
            batch,
            born_radius,
            monopole_prefactor,
            dipole_prefactor,
            self.gaussian_constant,
        )
        solvation_free_energy = reaction_free_energy(
            monopole_charges,
            atomic_dipoles,
            reaction_potential,
            reaction_field,
            batch,
            number_of_graphs,
        )
        return reaction_potential, reaction_field, solvation_free_energy
