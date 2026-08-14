"""ddCOSMO reaction-field *provider* (per-atom potential + field) for in-loop injection.

The reference-faithful tier of the swappable :class:`ReactionFieldProvider` interface. Where
:class:`~reaction_field.ddcosmo.DdcosmoReactionField` returns only the scalar ddCOSMO free energy
``G = 0.5 f(eps) <psi, X>`` (X = the induced surface-charge coefficients from the
domain-decomposition solve), a provider must return the per-atom reaction potential ``phi``
(conjugate to the atomic charge) and field ``E`` (conjugate to the atomic dipole) so the solvent
can be injected into PolarMACE's recursion and re-polarise the solute.

``G`` is an exact quadratic form in the solute multipoles ``M = (q, mu)`` (both enter ``psi`` and
the RHS ``phi`` linearly), so ``phi = dG/dq`` and ``E = dG/dmu`` are obtained by differentiating the
(pyscf-matched, tested) energy with autograd -- no reimplementation of ddCOSMO's analytic adjoint.
By Euler's theorem for a degree-2 homogeneous form, ``0.5 * sum_i (q_i phi_i + mu_i . E_i) == G``
exactly, which is the correctness gate. The atomic dipole is passed in Cartesian order here (the way
:class:`SolvatedPolarMACE` extracts it from the density); the wrapped module works in e3nn (y,z,x)
order, so it is permuted inside the grad tape.

Unlike the closed-form GB/GK providers this one is NOT TorchScript-scriptable (it calls
``torch.autograd.grad`` in forward) and is the most expensive tier (a batched surface solve per
recursion step). It is wrapped in ``enable_grad`` with a detach fallback so it works whether or not
the caller has grad enabled.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .ddcosmo import DdcosmoReactionField
from .reaction_potential import ReactionFieldProvider
from .screened_potential import dielectric_scaling_from_dielectric_constant


class DdcosmoReactionFieldProvider(ReactionFieldProvider):
    """ddCOSMO provider: per-atom reaction potential + field via autograd of the ddCOSMO energy."""

    def __init__(
        self,
        smearing_width: float,
        lebedev_order: int = 29,
        max_spherical_harmonic_order: int = 6,
        regularization_eta: float = 0.1,
        solve_ridge: float = 1.0e-8,
    ) -> None:
        super().__init__()
        # l=1 solute: the atomic dipoles both source the surface charges and feel the reaction
        # field, which is the whole point of the in-loop coupling.
        self.ddcosmo = DdcosmoReactionField(
            smearing_width=smearing_width,
            lebedev_order=lebedev_order,
            max_spherical_harmonic_order=max_spherical_harmonic_order,
            regularization_eta=regularization_eta,
            solve_ridge=solve_ridge,
            solute_multipole_max_l=1,
        )
        # SolvatedPolarMACE passes Cartesian (x, y, z) atomic dipoles; DdcosmoReactionField expects
        # e3nn (y, z, x) and permutes to Cartesian internally. Map Cartesian -> e3nn on the way in.
        self.register_buffer("cartesian_to_e3nn", torch.tensor([1, 2, 0], dtype=torch.long))

    def _energy(
        self,
        charges: Tensor,
        dipoles_cartesian: Tensor,
        positions: Tensor,
        atomic_numbers: Tensor,
        batch: Tensor,
        scaling: Tensor,
    ) -> Tensor:
        dipoles_e3nn = dipoles_cartesian.index_select(1, self.cartesian_to_e3nn)
        return self.ddcosmo(charges, positions, atomic_numbers, batch, scaling, dipoles_e3nn)

    def forward(
        self,
        monopole_charges: Tensor,
        atomic_dipoles: Tensor,
        positions: Tensor,
        atomic_numbers: Tensor,
        batch: Tensor,
        dielectric_constant: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scaling = dielectric_scaling_from_dielectric_constant(dielectric_constant)
        connected = torch.is_grad_enabled() and monopole_charges.requires_grad
        with torch.enable_grad():
            if connected:
                charges = monopole_charges
                dipoles = atomic_dipoles
            else:
                charges = monopole_charges.detach().requires_grad_(True)
                dipoles = atomic_dipoles.detach().requires_grad_(True)
            # NOTE: gradient checkpointing was tried here and REGRESSED both memory and speed --
            # the injected phi/E require create_graph=True (second-order), so the checkpoint must
            # retain the recompute graph rather than free activations, and it adds a recompute.
            energy = self._energy(
                charges, dipoles, positions, atomic_numbers, batch, scaling
            )  # [n_graphs], already scaled by f(eps)
            reaction_potential, reaction_field = torch.autograd.grad(
                energy.sum(),
                (charges, dipoles),
                create_graph=connected,
                retain_graph=True,
            )
        if not connected:
            energy = energy.detach()
        return reaction_potential, reaction_field, energy
