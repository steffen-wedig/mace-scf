"""Differentiable continuum reaction-field (implicit-solvation) modules for mace-scf.

Two one-shot, autograd-differentiable, batch-isolated, TorchScript-scriptable reaction-field
schemes operating on predicted atomic monopole charges + geometry + per-graph dielectric
scaling ``f(eps) = (eps-1)/eps``:

* :class:`ScreenedGeneralizedBornSolvation` -- the screened generalized-Born monopole scaffold
  / Exp-4 baseline (a permanent sanity check, not the product).
* :class:`DdcosmoReactionField` -- the production ddCOSMO model, matched to pyscf ddCOSMO / the
  C-PCM reference by cavity and dielectric scaling.

Shared, single-source helpers (the erf-screened Coulomb kernel reused from ``graph_longrange``,
the tensor dielectric scaling, the frozen cavity constants) live in
:mod:`reaction_field.screened_potential` and :mod:`reaction_field.constants`.

Everything is in eV / Angstrom / elementary-charge; reaction-field energies are negative for a
polar solute in a polar solvent and exactly 0 at ``eps == 1`` (gas).
"""

from __future__ import annotations

from .constants import (
    CAVITY_LEBEDEV_ORDER,
    HARTREE_IN_ELECTRON_VOLT,
    cavity_radius_angstrom,
    dielectric_scaling,
)
from .ddcosmo import DdcosmoReactionField
from .ddcosmo_provider import DdcosmoReactionFieldProvider
from .generalized_kirkwood import (
    GENERALIZED_KIRKWOOD_GAUSSIAN_CONSTANT,
    GeneralizedKirkwoodReactionField,
)
from .reaction_potential import (
    GeneralizedBornReactionField,
    ReactionFieldProvider,
    reaction_free_energy,
)
from .screened_born import ScreenedGeneralizedBornSolvation
from .screened_potential import (
    COULOMB_CONSTANT_EV_ANGSTROM,
    build_cavity_radius_lookup,
    compute_obc_born_radii,
    dielectric_scaling_from_dielectric_constant,
    generalized_born_descreening_factor,
    generalized_born_effective_distance,
    kirkwood_dielectric_factor,
    screened_coulomb_kernel,
    screened_gaussian_potential_at_points,
)

__all__ = [
    "ScreenedGeneralizedBornSolvation",
    "DdcosmoReactionField",
    "DdcosmoReactionFieldProvider",
    "ReactionFieldProvider",
    "GeneralizedBornReactionField",
    "GeneralizedKirkwoodReactionField",
    "GENERALIZED_KIRKWOOD_GAUSSIAN_CONSTANT",
    "reaction_free_energy",
    "dielectric_scaling",
    "dielectric_scaling_from_dielectric_constant",
    "kirkwood_dielectric_factor",
    "cavity_radius_angstrom",
    "build_cavity_radius_lookup",
    "compute_obc_born_radii",
    "screened_coulomb_kernel",
    "screened_gaussian_potential_at_points",
    "generalized_born_effective_distance",
    "generalized_born_descreening_factor",
    "COULOMB_CONSTANT_EV_ANGSTROM",
    "CAVITY_LEBEDEV_ORDER",
    "HARTREE_IN_ELECTRON_VOLT",
]
