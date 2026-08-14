"""Reaction-field *provider* interface + the monopole generalized-Born provider.

Where :mod:`reaction_field.screened_born` and :mod:`reaction_field.ddcosmo` return a single
scalar solvation energy (a *one-shot* correction added after the density is fixed), a
**provider** returns the per-atom reaction *potential* and *field* so the solvent can be injected
into a self-consistent (recursive) charge update -- the mechanism that lets the solute
re-polarise to the medium. Both continuum schemes are linear-response, so the free energy is the
symmetric quadratic form

    G = 1/2 * sum_i ( q_i * phi_i + mu_i . E_i )

where ``phi_i`` (conjugate to the monopole ``q_i``) is the reaction potential and ``E_i``
(conjugate to the atomic dipole ``mu_i``) is the reaction field. A provider therefore returns
``(reaction_potential[n_atoms], reaction_field[n_atoms, 3], solvation_free_energy[n_graphs])``.

Tiers implementing this interface (increasing fidelity, one-line swap in the model):
* :class:`GeneralizedBornReactionField` (here) -- monopole-only Born; ``E == 0`` by construction
  (no dipole coupling). The sanity baseline.
* :class:`~reaction_field.generalized_kirkwood.GeneralizedKirkwoodReactionField` -- the multipole
  (charge + atomic-dipole) extension of GB; the first *polarizable* tier.
* ddCOSMO surface model (future) -- the reference-faithful production tier.

Everything is eV / Angstrom / elementary-charge; ``eps == 1`` (gas) gives every dielectric factor
0, hence ``phi == 0``, ``E == 0``, and zero energy/gradient. All providers are pure tensor ops and
TorchScript-scriptable.
"""

from __future__ import annotations

import torch
from mace.tools.scatter import scatter_sum
from torch import Tensor

from .screened_potential import (
    COULOMB_CONSTANT_EV_ANGSTROM,
    build_cavity_radius_lookup,
    build_intramolecular_pairs,
    compute_obc_born_radii,
    generalized_born_effective_distance,
    kirkwood_dielectric_factor,
    screened_coulomb_kernel,
)


def reaction_free_energy(
    monopole_charges: Tensor,
    atomic_dipoles: Tensor,
    reaction_potential: Tensor,
    reaction_field: Tensor,
    batch: Tensor,
    num_graphs: int,
) -> Tensor:
    """Linear-response reaction free energy ``G = 1/2 sum_i (q_i phi_i + mu_i . E_i)`` per graph.

    Shared by every provider so the energy is one contraction of the same ``(phi, E)`` that gets
    injected into the charge update -- guaranteeing energy/field consistency. With a symmetric
    interaction (all tiers here), ``phi_i = dG/dq_i`` and ``E_i = dG/dmu_i`` hold identically.
    """
    per_atom = monopole_charges * reaction_potential + (atomic_dipoles * reaction_field).sum(
        dim=-1
    )
    return 0.5 * scatter_sum(src=per_atom, index=batch, dim=0, dim_size=num_graphs)


class ReactionFieldProvider(torch.nn.Module):
    """Interface: map the solute multipole density to the per-atom reaction potential + field.

    Concrete tiers implement :meth:`forward` with the signature

        forward(monopole_charges[n_atoms], atomic_dipoles[n_atoms, 3], positions[n_atoms, 3],
                atomic_numbers[n_atoms], batch[n_atoms], dielectric_constant[n_graphs])
            -> (reaction_potential[n_atoms], reaction_field[n_atoms, 3],
                solvation_free_energy[n_graphs])

    ``dielectric_constant`` is the raw per-graph ``eps`` (gas == 1.0); each tier applies its own
    order-dependent dielectric scaling internally.
    """

    def forward(
        self,
        monopole_charges: Tensor,
        atomic_dipoles: Tensor,
        positions: Tensor,
        atomic_numbers: Tensor,
        batch: Tensor,
        dielectric_constant: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class GeneralizedBornReactionField(ReactionFieldProvider):
    """Monopole screened generalized-Born provider (reaction potential only; ``E == 0``).

    The reaction potential is the exact derivative of the screened-GB energy,
    ``phi_i = -k f_0(eps) sum_j q_j K(f_GB(i,j))`` (self term ``j = i`` uses the Born radius),
    with ``K`` the erf-screened kernel and ``f_0 = (eps-1)/eps`` the Born dielectric factor. Being
    a pure monopole (Born) model it produces **no** field conjugate to the atomic dipole
    (``dG/dmu = 0``), so ``E == 0``: it drives charges but not dipoles. That is the whole point of
    the baseline -- the dipole coupling is what the generalized-Kirkwood tier adds. The returned
    energy equals :class:`~reaction_field.screened_born.ScreenedGeneralizedBornSolvation` exactly.
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
        number_of_atoms = positions.shape[0]
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
        monopole_scaling = kirkwood_dielectric_factor(dielectric_constant, 0)  # [n_graphs]
        monopole_scaling_per_atom = monopole_scaling.index_select(0, batch)  # [n_atoms]

        # Born self term: phi_i += -k f_0 q_i K(born_i).
        self_kernel = screened_coulomb_kernel(born_radius, self.smearing_width)
        reaction_potential = (
            -self.coulomb_constant * monopole_scaling_per_atom * monopole_charges * self_kernel
        )

        # Pair term over the block-diagonal intramolecular graph.
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
                4.0,
            )
            pair_kernel = screened_coulomb_kernel(effective_distance, self.smearing_width)
            pair_contribution = (
                -self.coulomb_constant
                * monopole_scaling_per_atom.index_select(0, receiver)
                * monopole_charges.index_select(0, sender)
                * pair_kernel
            )
            reaction_potential = reaction_potential + scatter_sum(
                src=pair_contribution, index=receiver, dim=0, dim_size=number_of_atoms
            )

        reaction_field = torch.zeros(
            (number_of_atoms, 3), dtype=positions.dtype, device=positions.device
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
