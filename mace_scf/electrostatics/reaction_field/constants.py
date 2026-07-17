"""Frozen physics-matching constants for the reaction-field (implicit-solvation) modules.

These values MUST stay identical to the reference generator's continuum settings so the
MACE reaction field and the gpu4pyscf C-PCM / pyscf ddCOSMO reference describe the *same*
cavity and the *same* dielectric scaling by construction (design_doc.md Section 3).

Source of truth: ``solvation_common.constants`` in the implicit_solvation project
(``/dais/u/wedigs/implicit_solvation/src/solvation_common/constants.py``), which reads the
values directly from the pinned ``pyscf.solvent.pcm`` C-PCM implementation. mace_scf must
ship independently of that project package, so the numeric values are duplicated here. Any
change to the reference protocol must change BOTH files.

The values below were copied verbatim from ``solvation_common.constants``:

* ``dielectric_scaling(eps) = (eps - 1) / eps`` -- the ``x = 0`` COSMO / C-PCM form
  (``pyscf.solvent.pcm.PCM.build``: ``f_epsilon = (epsilon - 1.) / epsilon``). Returns 0.0
  at ``eps == 1`` (gas, no reaction field) and 1.0 at ``eps -> inf`` (conductor limit).
* Cavity radius per atom = ``vdw_scale * modified_Bondi[Z] + r_probe`` with
  ``vdw_scale = 1.2`` (Q-Chem default) and ``r_probe = 0.0``. ``modified_Bondi`` is the Bondi
  van-der-Waals table with the hydrogen radius overridden to 1.10 Angstrom.
* Default Lebedev cavity grid order 29 (302 points per sphere).
"""

from __future__ import annotations

import math

# --- fundamental conversions (pyscf values, single source of truth) ---
# pyscf.data.radii.BOHR: 1 Bohr in Angstrom.
BOHR_IN_ANGSTROM: float = 0.52917721092
# CODATA Hartree energy in electron-volt (pyscf.data.nist.HARTREE2EV). Used only to relate the
# eV energies produced here to pyscf's Hartree reference energies in the validation tests.
HARTREE_IN_ELECTRON_VOLT: float = 27.211386245988

# --- C-PCM / ddCOSMO dielectric scaling (pyscf.solvent.pcm.PCM.build, method='C-PCM') ---
# f(eps) = (eps - 1) / (eps + x) evaluated at x = 0.
COSMO_SCALING_X: float = 0.0


def dielectric_scaling(dielectric_constant: float) -> float:
    """Scalar C-PCM / ddCOSMO dielectric scaling ``f(eps) = (eps - 1) / (eps + x)`` at x = 0.

    Returns 0.0 for the gas-phase limit ``eps == 1`` (no reaction field) and 1.0 for the
    conductor limit ``eps -> inf``. See :func:`reaction_field.screened_potential.
    dielectric_scaling_from_dielectric_constant` for the tensor-valued version used inside the
    model.
    """
    if math.isinf(dielectric_constant):
        return 1.0
    return (dielectric_constant - 1.0) / (dielectric_constant + COSMO_SCALING_X)


# --- cavity construction (pyscf.solvent.pcm defaults) ---
CAVITY_VDW_SCALE: float = 1.2  # pcm.PCM.vdw_scale default (Q-Chem convention)
CAVITY_PROBE_RADIUS_ANGSTROM: float = 0.0  # pcm.PCM.r_probe default
CAVITY_LEBEDEV_ORDER: int = 29  # pcm.PCM.lebedev_order default -> 302 points per sphere

# Lebedev order -> number of angular grid points (pyscf.dft.gen_grid.LEBEDEV_ORDER).
LEBEDEV_ORDER_TO_POINTS: dict[int, int] = {
    11: 50,
    17: 110,
    23: 194,
    29: 302,
    35: 434,
}

# modified_Bondi van-der-Waals radii in Angstrom (pyscf.data.radii.VDW * BOHR, with H -> 1.10).
# Frozen for the molecule-set elements (H, C, N, O, F, S, Cl). An unknown element must raise
# rather than silently mismatch the reference cavity.
MODIFIED_BONDI_RADII_ANGSTROM: dict[int, float] = {
    1: 1.10,  # H (modified from Bondi 1.20)
    6: 1.70,  # C
    7: 1.55,  # N
    8: 1.52,  # O
    9: 1.47,  # F
    16: 1.80,  # S
    17: 1.75,  # Cl
}


def cavity_radius_angstrom(atomic_number: int) -> float:
    """Cavity sphere radius for one atom: ``vdw_scale * modified_Bondi[Z] + r_probe``.

    Matches ``pcm.PCM.build``'s ``radii_table = vdw_scale * modified_Bondi + r_probe/BOHR``
    (that expression is in Bohr; here everything is Angstrom). Raises ``KeyError`` for an
    element absent from :data:`MODIFIED_BONDI_RADII_ANGSTROM` so a cavity mismatch surfaces
    loudly instead of defaulting.
    """
    if atomic_number not in MODIFIED_BONDI_RADII_ANGSTROM:
        raise KeyError(
            f"No frozen cavity radius for atomic number {atomic_number}; add it to "
            "MODIFIED_BONDI_RADII_ANGSTROM after confirming the pyscf reference value."
        )
    base = MODIFIED_BONDI_RADII_ANGSTROM[atomic_number]
    return CAVITY_VDW_SCALE * base + CAVITY_PROBE_RADIUS_ANGSTROM


def highest_supported_atomic_number() -> int:
    """Largest atomic number with a frozen cavity radius (used to size lookup buffers)."""
    return max(MODIFIED_BONDI_RADII_ANGSTROM)
