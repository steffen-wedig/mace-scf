"""ddCOSMO reaction field -- the production continuum model.

Domain-decomposition COSMO (Cances/Maday/Stamm; Lipparini et al.) in a per-atom real
spherical-harmonic basis, matched to ``pyscf.solvent.ddcosmo`` by construction. The linear
system is solved DIRECTLY (``torch.linalg.solve``) -- the per-molecule matrix is small
(``n_atoms * (lmax+1)**2``) so no iterative solver / custom adjoint is needed, and the whole
functional is one-shot autograd-differentiable.

Working equations (see pyscf.solvent.ddcosmo make_L / make_fi / make_phi / _get_vind and
JCTC 9, 3637 / JCP 141, 184108), specialised to an atomic *monopole* solute (l = 0 charges):

    cavity: ball of radius r_vdw[a] = 1.2 * modified_Bondi[Z_a] around each atom
    fi[a, n]      = sum_{b != a} chi_eta(|r_vdw[a] s_n + R_a - R_b| / r_vdw[b])   (burial)
    ui[a, n]      = max(1 - fi[a, n], 0)                                          (exposure)
    V(cav)        = k * sum_b q_b * erf(|cav - R_b| / (sqrt2 * width)) / |cav - R_b|  (solute)
    phi[a, lm]    = - sum_n w_n Y_lm(s_n) ui[a, n] V(cav_{a,n})                   (RHS)
    L (block)     = diag 4pi/(2l+1)/r_vdw  -  neighbour coupling via solid harmonics
    X             = L^{-1} phi
    psi[a, 0]     = sqrt(4pi) q_a / r_vdw[a]    (monopole solute; psi[a, l>0] = 0)
    dG_ddcosmo    = 0.5 * f(eps) * sum_{a,lm} psi[a,lm] X[a,lm]

Units: eV / Angstrom / elementary-charge; the Coulomb constant enters once through ``V``. With
geometry in Angstrom this reproduces pyscf's Hartree energy times HARTREE_IN_ELECTRON_VOLT
(the Coulomb constant equals BOHR_IN_ANGSTROM * HARTREE_IN_ELECTRON_VOLT). ``dG_ddcosmo < 0``
for a polar solute in a polar solvent, and is exactly 0 at ``eps == 1`` (f = 0).

Solute representation: atomic monopole charges smeared as Gaussians of width ``smearing_width``
enter ``V`` (RHS); the energy contraction uses the exact monopole ``psi`` (a centred spherical
Gaussian well inside its sphere has the same l = 0 moment as a point charge and zero higher
moments). Both agree with pyscf point charges as ``smearing_width -> 0``. Higher-l solute
multipoles (dipole psi terms) are a documented future extension; the l = 0 path is complete.

Batch isolation: the surface solve is done per molecule (one ``torch.linalg.solve`` per graph),
so molecule A's cavity never sees molecule B regardless of how the batch is laid out in space.
"""

from __future__ import annotations

import math

import numpy
import torch
from torch import Tensor

from .constants import CAVITY_LEBEDEV_ORDER
from .screened_potential import (
    COULOMB_CONSTANT_EV_ANGSTROM,
    build_cavity_radius_lookup,
    point_target_effective_width,
    screened_coulomb_kernel,
)


def _monomial_exponents(angular_momentum: int) -> list[tuple[int, int, int]]:
    """Cartesian monomial (lx, ly, lz) ordering used by ``pyscf.symm.sph.multipoles``."""
    exponents = []
    for exponent_x in reversed(range(0, angular_momentum + 1)):
        for exponent_y in reversed(range(0, angular_momentum - exponent_x + 1)):
            exponents.append((exponent_x, exponent_y, angular_momentum - exponent_x - exponent_y))
    return exponents


class DdcosmoReactionField(torch.nn.Module):
    """ddCOSMO reaction-field energy (per graph) for an atomic monopole solute.

    forward(charges, positions, atomic_numbers, batch, dielectric_scaling) -> ``[n_graphs]``

    Args of ``forward``:
        charges: ``[n_atoms]`` predicted atomic monopole charges (elementary charge).
        positions: ``[n_atoms, 3]`` Angstrom.
        atomic_numbers: ``[n_atoms]`` long; used to look up the cavity radius per atom.
        batch: ``[n_atoms]`` long; graph id of each atom (per-molecule surface solve).
        dielectric_scaling: ``[n_graphs]`` per-graph ``f(eps) = (eps-1)/eps``. Pass 0.0 at
            ``eps == 1`` -> the returned energy and its gradient are exactly 0.

    Returns:
        ``[n_graphs]`` reaction-field free energy in eV.

    Construction:
        smearing_width: Gaussian width of the solute density (Angstrom; design default 1.5).
        lebedev_order: angular grid order per sphere (default 29 -> 302 points), matched to the
            C-PCM/ddCOSMO reference cavity grid.
        max_spherical_harmonic_order: ``lmax`` of the surface harmonic basis (default 6, the
            pyscf ddCOSMO default). Larger is more accurate and larger matrices.
        regularization_eta: ddCOSMO switching-region width ``eta`` (default 0.1).
        solve_ridge: small diagonal ridge added to the COSMO matrix before the direct solve,
            guarding against a near-singular cavity.

    Construction imports pyscf lazily to precompute the (frozen) Lebedev grid, real spherical
    harmonics, and cart2sph transforms as registered buffers; inference/scripting needs only
    those buffers (pyscf is not touched in ``forward``).
    """

    def __init__(
        self,
        smearing_width: float,
        lebedev_order: int = CAVITY_LEBEDEV_ORDER,
        max_spherical_harmonic_order: int = 6,
        regularization_eta: float = 0.1,
        solve_ridge: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.smearing_width = float(smearing_width)
        self.point_effective_width = point_target_effective_width(self.smearing_width)
        self.max_spherical_harmonic_order = int(max_spherical_harmonic_order)
        self.regularization_eta = float(regularization_eta)
        self.solve_ridge = float(solve_ridge)
        self.coulomb_constant = COULOMB_CONSTANT_EV_ANGSTROM
        self.number_of_lm = (self.max_spherical_harmonic_order + 1) ** 2
        self.sqrt_four_pi = math.sqrt(4.0 * math.pi)

        # --- lazily import pyscf only for the one-time frozen-grid precomputation ---
        from pyscf import gto
        from pyscf.solvent import ddcosmo as pyscf_ddcosmo
        from pyscf.symm import sph

        lebedev_coordinates, lebedev_weights = pyscf_ddcosmo.make_grids_one_sphere(lebedev_order)
        self.number_of_grid_points = int(lebedev_coordinates.shape[0])

        real_spherical_harmonics = numpy.vstack(
            sph.real_sph_vec(lebedev_coordinates, self.max_spherical_harmonic_order, True)
        )  # [number_of_lm, number_of_grid_points]

        # Cartesian-monomial -> real-solid-harmonic transform, laid out block-diagonally so a
        # single matmul yields all solid harmonics up to lmax (matches sph.multipoles exactly).
        exponents_x: list[int] = []
        exponents_y: list[int] = []
        exponents_z: list[int] = []
        cart2sph_blocks = []
        lm_angular_momentum: list[int] = []
        for angular_momentum in range(self.max_spherical_harmonic_order + 1):
            exponents = _monomial_exponents(angular_momentum)
            for exponent_x, exponent_y, exponent_z in exponents:
                exponents_x.append(exponent_x)
                exponents_y.append(exponent_y)
                exponents_z.append(exponent_z)
            number_of_cartesians = len(exponents)
            cart2sph_blocks.append(
                gto.cart2sph(angular_momentum, numpy.eye(number_of_cartesians))
            )  # [number_of_cartesians, 2l+1]
            lm_angular_momentum.extend([angular_momentum] * (2 * angular_momentum + 1))

        number_of_cartesians_total = len(exponents_x)
        cart2sph_blockdiagonal = numpy.zeros((number_of_cartesians_total, self.number_of_lm))
        cart_offset = 0
        lm_offset = 0
        for block in cart2sph_blocks:
            rows, cols = block.shape
            cart2sph_blockdiagonal[
                cart_offset : cart_offset + rows, lm_offset : lm_offset + cols
            ] = block
            cart_offset += rows
            lm_offset += cols

        lm_angular_momentum_array = numpy.asarray(lm_angular_momentum)
        four_pi_over_two_l_plus_one = 4.0 * math.pi / (2.0 * lm_angular_momentum_array + 1.0)

        default_dtype = torch.get_default_dtype()
        self.register_buffer(
            "lebedev_coordinates", torch.tensor(lebedev_coordinates, dtype=default_dtype)
        )
        self.register_buffer(
            "lebedev_weights", torch.tensor(lebedev_weights, dtype=default_dtype)
        )
        self.register_buffer(
            "real_spherical_harmonics",
            torch.tensor(real_spherical_harmonics, dtype=default_dtype),
        )
        self.register_buffer(
            "cart2sph_blockdiagonal",
            torch.tensor(cart2sph_blockdiagonal, dtype=default_dtype),
        )
        self.register_buffer("monomial_exponent_x", torch.tensor(exponents_x, dtype=torch.long))
        self.register_buffer("monomial_exponent_y", torch.tensor(exponents_y, dtype=torch.long))
        self.register_buffer("monomial_exponent_z", torch.tensor(exponents_z, dtype=torch.long))
        self.register_buffer(
            "lm_angular_momentum", torch.tensor(lm_angular_momentum_array, dtype=torch.long)
        )
        self.register_buffer(
            "four_pi_over_two_l_plus_one",
            torch.tensor(four_pi_over_two_l_plus_one, dtype=default_dtype),
        )
        self.register_buffer("cavity_radius_lookup", build_cavity_radius_lookup(default_dtype))

    def _regularize_switching(self, scaled_distance: Tensor) -> Tensor:
        """ddCOSMO switching function ``chi_eta`` (pyscf.solvent.ddcosmo.regularize_xt).

        1 for ``t <= 1 - eta`` (buried), a smooth polynomial for ``1 - eta < t < 1``, 0 for
        ``t >= 1`` (outside the neighbour sphere).
        """
        eta = self.regularization_eta
        one_minus_eta = 1.0 - eta
        inner = scaled_distance <= one_minus_eta
        on_shell = (scaled_distance > one_minus_eta) & (scaled_distance < 1.0)
        polynomial = (1.0 / eta**5) * (1.0 - scaled_distance) ** 3 * (
            6.0 * scaled_distance * scaled_distance
            + (15.0 * eta - 12.0) * scaled_distance
            + 10.0 * eta * eta
            - 15.0 * eta
            + 6.0
        )
        switching = torch.zeros_like(scaled_distance)
        switching = torch.where(inner, torch.ones_like(scaled_distance), switching)
        switching = torch.where(on_shell, polynomial, switching)
        return switching

    def _solid_harmonics(self, vectors: Tensor) -> Tensor:
        """Real regular solid harmonics ``R_lm(v)`` up to lmax; ``[number_of_lm, n_points]``.

        Exactly reproduces ``pyscf.symm.sph.multipoles`` via the cart2sph transform. Powers are
        built by cumulative multiplication (never ``pow`` of a negative base, which would NaN).
        """
        number_of_points = vectors.shape[0]
        component_x = vectors[:, 0]
        component_y = vectors[:, 1]
        component_z = vectors[:, 2]
        power_tables_x = [torch.ones(number_of_points, dtype=vectors.dtype, device=vectors.device)]
        power_tables_y = [torch.ones(number_of_points, dtype=vectors.dtype, device=vectors.device)]
        power_tables_z = [torch.ones(number_of_points, dtype=vectors.dtype, device=vectors.device)]
        for _ in range(self.max_spherical_harmonic_order):
            power_tables_x.append(power_tables_x[-1] * component_x)
            power_tables_y.append(power_tables_y[-1] * component_y)
            power_tables_z.append(power_tables_z[-1] * component_z)
        powers_x = torch.stack(power_tables_x, dim=0)  # [lmax+1, n_points]
        powers_y = torch.stack(power_tables_y, dim=0)
        powers_z = torch.stack(power_tables_z, dim=0)
        monomials = (
            powers_x.index_select(0, self.monomial_exponent_x)
            * powers_y.index_select(0, self.monomial_exponent_y)
            * powers_z.index_select(0, self.monomial_exponent_z)
        )  # [number_of_cartesians_total, n_points]
        return self.cart2sph_blockdiagonal.t() @ monomials  # [number_of_lm, n_points]

    def _graph_reaction_field_energy(
        self,
        positions: Tensor,
        charges: Tensor,
        cavity_radii: Tensor,
    ) -> Tensor:
        """f(eps)-independent ddCOSMO energy ``0.5 * <psi, L^{-1} phi>`` for one molecule."""
        number_of_atoms = positions.shape[0]
        number_of_grid_points = self.number_of_grid_points
        number_of_lm = self.number_of_lm

        # cavity grid points: [n_atoms, n_grid, 3]
        cavity_points = (
            cavity_radii.view(number_of_atoms, 1, 1)
            * self.lebedev_coordinates.view(1, number_of_grid_points, 3)
            + positions.view(number_of_atoms, 1, 3)
        )

        # --- burial fi and exposure ui ---
        burial = torch.zeros(
            number_of_atoms, number_of_grid_points, dtype=positions.dtype, device=positions.device
        )
        for atom_index in range(number_of_atoms):
            for neighbour_index in range(number_of_atoms):
                if neighbour_index == atom_index:
                    continue
                relative = cavity_points[atom_index] - positions[neighbour_index]
                scaled_distance = (
                    torch.linalg.norm(relative, dim=1) / cavity_radii[neighbour_index]
                )
                burial[atom_index] = burial[atom_index] + self._regularize_switching(
                    scaled_distance
                )
        burial = torch.where(burial < 1.0e-20, torch.zeros_like(burial), burial)
        exposure = torch.clamp(1.0 - burial, min=0.0)

        # --- solute potential on the cavity grid and its harmonic projection phi ---
        displacement = cavity_points.unsqueeze(2) - positions.view(1, 1, number_of_atoms, 3)
        distance = torch.linalg.norm(displacement, dim=3)  # [n_atoms, n_grid, n_atoms]
        kernel = screened_coulomb_kernel(distance, self.point_effective_width)
        potential = self.coulomb_constant * (
            charges.view(1, 1, number_of_atoms) * kernel
        ).sum(dim=2)  # [n_atoms, n_grid]
        weighted = self.lebedev_weights.view(1, number_of_grid_points) * exposure * potential
        right_hand_side = -(weighted @ self.real_spherical_harmonics.t())  # [n_atoms, number_of_lm]

        # --- ddCOSMO matrix L ---
        four_pi_over_two_l_plus_one = self.four_pi_over_two_l_plus_one
        angular_momentum_float = self.lm_angular_momentum.to(positions.dtype)
        diagonal_values = four_pi_over_two_l_plus_one.view(1, number_of_lm) / cavity_radii.view(
            number_of_atoms, 1
        )  # [n_atoms, number_of_lm]
        cosmo_matrix = torch.diag(diagonal_values.reshape(-1))  # [n_atoms*nlm, n_atoms*nlm]

        for atom_index in range(number_of_atoms):
            partition_weights = self.lebedev_weights / torch.clamp(burial[atom_index], min=1.0)
            for source_index in range(number_of_atoms):
                if source_index == atom_index:
                    continue
                relative = cavity_points[atom_index] - positions[source_index]  # [n_grid, 3]
                scaled_distance = torch.linalg.norm(relative, dim=1) / cavity_radii[source_index]
                grid_weights = self._regularize_switching(scaled_distance) * partition_weights
                solid_harmonics = self._solid_harmonics(relative)  # [number_of_lm, n_grid]
                source_factor = four_pi_over_two_l_plus_one / cavity_radii[source_index] ** (
                    angular_momentum_float + 1.0
                )  # [number_of_lm]
                coupling = (
                    self.real_spherical_harmonics * grid_weights.view(1, number_of_grid_points)
                ) @ solid_harmonics.t()  # [number_of_lm, number_of_lm]
                block = -source_factor.view(1, number_of_lm) * coupling
                row_start = atom_index * number_of_lm
                source_start = source_index * number_of_lm
                cosmo_matrix[
                    row_start : row_start + number_of_lm,
                    source_start : source_start + number_of_lm,
                ] = (
                    cosmo_matrix[
                        row_start : row_start + number_of_lm,
                        source_start : source_start + number_of_lm,
                    ]
                    + block
                )

        # small ridge guards a near-singular cavity before the direct solve
        matrix_dimension = number_of_atoms * number_of_lm
        cosmo_matrix = cosmo_matrix + self.solve_ridge * torch.eye(
            matrix_dimension, dtype=positions.dtype, device=positions.device
        )

        surface_coefficients = torch.linalg.solve(
            cosmo_matrix, right_hand_side.reshape(-1)
        ).reshape(number_of_atoms, number_of_lm)

        # --- monopole source projection psi and the energy contraction ---
        source_projection = torch.zeros(
            number_of_atoms, number_of_lm, dtype=positions.dtype, device=positions.device
        )
        source_projection[:, 0] = self.sqrt_four_pi * charges / cavity_radii
        return 0.5 * (source_projection * surface_coefficients).sum()

    def forward(
        self,
        charges: Tensor,
        positions: Tensor,
        atomic_numbers: Tensor,
        batch: Tensor,
        dielectric_scaling: Tensor,
    ) -> Tensor:
        number_of_graphs = int(dielectric_scaling.shape[0])
        cavity_radii = self.cavity_radius_lookup.index_select(0, atomic_numbers)

        energies = torch.zeros(
            number_of_graphs, dtype=positions.dtype, device=positions.device
        )
        for graph_index in range(number_of_graphs):
            atom_indices = torch.nonzero(batch == graph_index).squeeze(-1)
            if atom_indices.numel() == 0:
                continue
            graph_energy = self._graph_reaction_field_energy(
                positions.index_select(0, atom_indices),
                charges.index_select(0, atom_indices),
                cavity_radii.index_select(0, atom_indices),
            )
            energies[graph_index] = graph_energy

        return energies * dielectric_scaling
