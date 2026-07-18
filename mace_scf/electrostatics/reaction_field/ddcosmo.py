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

    def _batched_reaction_field_energy(
        self,
        positions: Tensor,
        charges: Tensor,
        cavity_radii: Tensor,
        atom_valid: Tensor,
    ) -> Tensor:
        """f(eps)-independent ddCOSMO energy ``0.5 * <psi, L^{-1} phi>`` for a padded batch.

        Inputs are dense ``[n_graphs, max_atoms, ...]`` tensors with ``atom_valid`` marking the
        real atoms (padding atoms carry charge 0 and cavity radius 1). Every molecule keeps its
        own surface solve: the pair mask (``valid_a & valid_b & a != b``) zeros all couplings that
        touch a padding atom, so each molecule's block is exactly the single-molecule ddCOSMO
        matrix, the padding rows decouple into a trivial diagonal block, and a padding atom's zero
        charge makes its ``psi`` -- and hence its energy contribution -- exactly zero. The result
        is therefore identical (up to floating point) to solving each molecule separately, but as
        one batched ``torch.linalg.solve`` -- which is what lets a larger batch actually speed
        ddCOSMO up on the GPU (small molecules otherwise underutilise it one solve at a time).

        Fully vectorised over both atom pairs and graphs; TorchScript-scriptable.
        """
        number_of_graphs = positions.shape[0]
        max_atoms = positions.shape[1]
        number_of_grid_points = self.number_of_grid_points
        number_of_lm = self.number_of_lm

        # cavity grid points: [n_graphs, max_atoms, n_grid, 3]
        cavity_points = (
            cavity_radii.view(number_of_graphs, max_atoms, 1, 1)
            * self.lebedev_coordinates.view(1, 1, number_of_grid_points, 3)
            + positions.view(number_of_graphs, max_atoms, 1, 3)
        )

        # Every cavity point of atom a relative to every atom b: [n_graphs, a, n_grid, b, 3].
        displacement = cavity_points.unsqueeze(3) - positions.view(
            number_of_graphs, 1, 1, max_atoms, 3
        )
        distance = torch.linalg.norm(displacement, dim=4)  # [n_graphs, a, n_grid, b]

        # Valid ordered atom pairs within a molecule (a != b, both real). Reproduces the pair
        # loops' ``b != a`` skip AND excludes padding atoms so molecules stay isolated.
        not_self = ~torch.eye(max_atoms, dtype=torch.bool, device=positions.device)
        valid_pair = (
            atom_valid.view(number_of_graphs, max_atoms, 1)
            & atom_valid.view(number_of_graphs, 1, max_atoms)
            & not_self.view(1, max_atoms, max_atoms)
        ).to(positions.dtype)  # [n_graphs, a, b]

        # --- burial fi and exposure ui ---
        scaled_distance = distance / cavity_radii.view(number_of_graphs, 1, 1, max_atoms)
        switching = self._regularize_switching(scaled_distance)  # [n_graphs, a, n_grid, b]
        switching = switching * valid_pair.view(number_of_graphs, max_atoms, 1, max_atoms)
        burial = switching.sum(dim=3)  # [n_graphs, a, n_grid]
        burial = torch.where(burial < 1.0e-20, torch.zeros_like(burial), burial)
        exposure = torch.clamp(1.0 - burial, min=0.0)

        # --- solute potential on the cavity grid and its harmonic projection phi ---
        # (padding sources carry charge 0, so summing over all b needs no source mask here.)
        kernel = screened_coulomb_kernel(distance, self.point_effective_width)
        potential = self.coulomb_constant * (
            charges.view(number_of_graphs, 1, 1, max_atoms) * kernel
        ).sum(dim=3)  # [n_graphs, a, n_grid]
        weighted = (
            self.lebedev_weights.view(1, 1, number_of_grid_points) * exposure * potential
        )
        right_hand_side = -torch.matmul(
            weighted, self.real_spherical_harmonics.t()
        )  # [n_graphs, a, number_of_lm]

        # --- ddCOSMO matrix L ---
        four_pi_over_two_l_plus_one = self.four_pi_over_two_l_plus_one
        angular_momentum_float = self.lm_angular_momentum.to(positions.dtype)
        diagonal_values = four_pi_over_two_l_plus_one.view(
            1, 1, number_of_lm
        ) / cavity_radii.view(number_of_graphs, max_atoms, 1)  # [n_graphs, a, number_of_lm]

        partition_weights = self.lebedev_weights.view(
            1, 1, number_of_grid_points
        ) / torch.clamp(burial, min=1.0)  # [n_graphs, a, n_grid]
        grid_weights = switching * partition_weights.view(
            number_of_graphs, max_atoms, number_of_grid_points, 1
        )  # [n_graphs, a, n_grid, b] (off-diagonal / padding already masked via ``switching``)

        # Source solid harmonics R_lm(cavity_point_a - R_b) for every (graph, a, grid, b): flatten
        # to reuse the tested [n_points, 3] -> [nlm, n_points] helper, then restore the layout.
        source_solid_harmonics = self._solid_harmonics(displacement.reshape(-1, 3)).reshape(
            number_of_lm, number_of_graphs, max_atoms, number_of_grid_points, max_atoms
        )  # [nlm_source, n_graphs, a, n_grid, b]
        weighted_source = source_solid_harmonics * grid_weights.view(
            1, number_of_graphs, max_atoms, number_of_grid_points, max_atoms
        )
        # coupling[g, a, b, t, s] = sum_grid real_sph[t, grid] * weighted_source[s, g, a, grid, b]
        weighted_source = weighted_source.permute(
            1, 2, 4, 0, 3
        )  # [n_graphs, a, b, nlm_source, n_grid]
        coupling = torch.matmul(
            self.real_spherical_harmonics.view(1, 1, 1, number_of_lm, number_of_grid_points),
            weighted_source.transpose(-1, -2),
        )  # [n_graphs, a, b, nlm_target, nlm_source]

        source_factor = four_pi_over_two_l_plus_one.view(
            1, 1, number_of_lm
        ) / cavity_radii.view(number_of_graphs, max_atoms, 1) ** (
            angular_momentum_float.view(1, 1, number_of_lm) + 1.0
        )  # [n_graphs, b, nlm_source]
        blocks = -source_factor.view(number_of_graphs, 1, max_atoms, 1, number_of_lm) * coupling
        # place block (a, b) at rows a*nlm.., cols b*nlm.. ; self / padding blocks are zero.
        neighbour_matrix = blocks.permute(0, 1, 3, 2, 4).reshape(
            number_of_graphs, max_atoms * number_of_lm, max_atoms * number_of_lm
        )

        matrix_dimension = max_atoms * number_of_lm
        identity = torch.eye(
            matrix_dimension, dtype=positions.dtype, device=positions.device
        ).view(1, matrix_dimension, matrix_dimension)
        cosmo_matrix = (
            neighbour_matrix
            + torch.diag_embed(diagonal_values.reshape(number_of_graphs, matrix_dimension))
            + self.solve_ridge * identity
        )

        surface_coefficients = torch.linalg.solve(
            cosmo_matrix, right_hand_side.reshape(number_of_graphs, matrix_dimension, 1)
        ).reshape(number_of_graphs, max_atoms, number_of_lm)

        # --- monopole source projection psi and the energy contraction (padding psi == 0) ---
        source_projection = torch.zeros(
            number_of_graphs,
            max_atoms,
            number_of_lm,
            dtype=positions.dtype,
            device=positions.device,
        )
        source_projection[:, :, 0] = self.sqrt_four_pi * charges / cavity_radii
        energy = 0.5 * (source_projection * surface_coefficients).reshape(
            number_of_graphs, -1
        ).sum(dim=1)
        return energy  # [n_graphs]

    def forward(
        self,
        charges: Tensor,
        positions: Tensor,
        atomic_numbers: Tensor,
        batch: Tensor,
        dielectric_scaling: Tensor,
    ) -> Tensor:
        number_of_graphs = int(dielectric_scaling.shape[0])
        number_of_atoms = positions.shape[0]
        if number_of_atoms == 0:
            return torch.zeros(
                number_of_graphs, dtype=positions.dtype, device=positions.device
            )
        cavity_radii = self.cavity_radius_lookup.index_select(0, atomic_numbers)

        # Pack the ragged batch into a dense [n_graphs, max_atoms, ...] layout. Assumes the atoms
        # of each graph are contiguous and in graph order (the standard mace/torch-geometric
        # batch layout); intra-graph position = global index - graph offset.
        counts = torch.zeros(number_of_graphs, dtype=torch.long, device=positions.device)
        counts = counts.scatter_add(
            0, batch, torch.ones(number_of_atoms, dtype=torch.long, device=positions.device)
        )
        max_atoms = int(counts.max())
        offsets = torch.cumsum(counts, dim=0) - counts  # exclusive prefix sum, [n_graphs]
        intra_index = torch.arange(
            number_of_atoms, device=positions.device
        ) - offsets.index_select(0, batch)
        linear_index = batch * max_atoms + intra_index  # [n_atoms] -> slot in the padded layout

        padded_size = number_of_graphs * max_atoms
        padded_positions = torch.zeros(
            padded_size, 3, dtype=positions.dtype, device=positions.device
        ).index_copy(0, linear_index, positions).view(number_of_graphs, max_atoms, 3)
        padded_charges = torch.zeros(
            padded_size, dtype=positions.dtype, device=positions.device
        ).index_copy(0, linear_index, charges).view(number_of_graphs, max_atoms)
        # Padding radius 1 (not 0) keeps 1/r and r**l finite; padding entries are masked out.
        padded_radii = torch.ones(
            padded_size, dtype=positions.dtype, device=positions.device
        ).index_copy(0, linear_index, cavity_radii).view(number_of_graphs, max_atoms)
        atom_valid = torch.arange(
            max_atoms, device=positions.device
        ).view(1, max_atoms) < counts.view(number_of_graphs, 1)  # [n_graphs, max_atoms]

        energies = self._batched_reaction_field_energy(
            padded_positions, padded_charges, padded_radii, atom_valid
        )
        return energies * dielectric_scaling
