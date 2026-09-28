"""TorchSim model interface for mace_scf local-source models (LocalSplitCharges).

Drop-in location upstream: ``mace_scf/calculators/torchsim.py``. The module depends only
on the public APIs of ``torch``, ``mace`` (``mace.tools``), ``mace_scf`` (a loaded
``LocalSplitCharges`` checkpoint) and ``torch_sim`` (``ModelInterface``, ``SimState``,
``transforms``, ``neighbors``). It is the batched counterpart of
``mace_scf.calculators.localsources.MACELocalSplitCharges``: that calculator evaluates one
``ase.Atoms`` through ``ExtAtomicData`` + a matscipy neighbour list on the host; this
wrapper evaluates a whole ``torch_sim.SimState`` (many systems, ragged sizes and cells)
in one forward, with the neighbour list built on the GPU.

Conventions (checked against ``MACELocalSplitCharges`` to ~1e-12 in float64):

* Cell. ``SimState.cell`` is column-vector; the model gets ``state.row_vector_cell``
  (ASE layout), and the TorchSim neighbour list's integer image shifts are turned into
  Cartesian shifts with that same row-vector cell, as ``ExtAtomicData`` does
  (``shifts = unit_shifts @ cell``).
* Stress. The model's ``stress`` is ``(1/V) dE/d(strain)`` in eV/Å^3 -- the ASE sign
  convention (positive = tensile) -- which is also TorchSim's (its barostats use
  ``P_virial = -Tr(stress)/3``), so it is returned unchanged, as 3x3 per system.
* Non-periodic systems. ``mace_scf.data.neighborhood.get_neighborhood`` replaces the cell
  of an ``pbc = (F, F, F)`` system by a nominal box (diagonal >= 30 Å, or a cube of twice
  the largest extent if that is too small). The model reads that box (volume and, with
  ``pbc_handling != "realspace"``, the k-space sum), so :func:`nonperiodic_cell` mirrors
  the rule exactly. TorchSim's ``pbc`` is global, so a batch is either all periodic or
  all non-periodic.
* Inputs the calculator takes from ``atoms`` are taken from state extras: per-atom formal
  charges from ``state.formal_charges`` (ASE: ``atoms.arrays["charges"]``) and the
  per-system external field from ``state.external_E_field`` (ASE:
  ``atoms.info["external_field"]``); both default to zero, as in ``ExtAtomicData``. Head:
  the model's first head, as in the calculator, unless ``head`` is given.

Example::

    import torch, torch_sim as ts
    from mace_scf.calculators.torchsim import LocalSplitChargesTorchSimModel

    model = LocalSplitChargesTorchSimModel("lsc.model", device="cuda",
                                           dtype=torch.float64, pbc_handling="pbc")
    state = ts.io.atoms_to_state(list_of_atoms, model.device, model.dtype)
    out = model(state)          # energy [S], forces [N, 3], stress [S, 3, 3]
    final = ts.integrate(state, model, integrator=ts.Integrator.nvt_vrescale,
                         n_steps=1000, temperature=300.0, timestep=0.0005)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
from mace.tools import atomic_numbers_to_indices, utils

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl

    _TORCHSIM_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - exercised only without torch-sim
    ts = None  # type: ignore[assignment]
    torchsim_nl = None  # type: ignore[assignment]
    _TORCHSIM_IMPORT_ERROR = exc

    class ModelInterface(torch.nn.Module):  # type: ignore[no-redef]
        """Fallback base class when torch-sim is not installed."""


__all__ = ["LocalSplitChargesTorchSimModel", "nonperiodic_cell"]

NONPERIODIC_MINIMUM_BOX_ANGSTROM = 30.0
"""The minimum nominal box edge ``get_neighborhood`` assigns to a non-periodic system."""


def nonperiodic_cell(
    row_vector_cell: torch.Tensor, positions: torch.Tensor, system_idx: torch.Tensor
) -> torch.Tensor:
    """The nominal cell ``mace_scf.data.neighborhood.get_neighborhood`` gives each
    non-periodic system, batched: a zero cell becomes the identity, every diagonal
    element is raised to at least 30 Å, and if any diagonal is still shorter than twice
    the system's extent along that axis the cell becomes a cube of twice the largest
    extent. Off-diagonal elements are kept, as there.

    Args:
        row_vector_cell: [S, 3, 3] cells in row-vector (ASE) layout.
        positions: [N, 3] positions.
        system_idx: [N] system index of every atom.
    """
    n_systems = row_vector_cell.shape[0]
    cell = row_vector_cell.detach().clone()
    zero = (cell == 0).all(dim=(1, 2))
    eye = torch.eye(3, dtype=cell.dtype, device=cell.device)
    cell[zero] = eye
    diagonal = torch.diagonal(cell, dim1=-2, dim2=-1).clamp(
        min=NONPERIODIC_MINIMUM_BOX_ANGSTROM
    )
    index = system_idx.view(-1, 1).expand(-1, 3)
    upper = torch.full((n_systems, 3), -torch.inf, dtype=cell.dtype, device=cell.device)
    lower = torch.full((n_systems, 3), torch.inf, dtype=cell.dtype, device=cell.device)
    pos = positions.detach()
    upper = upper.scatter_reduce(0, index, pos, reduce="amax")
    lower = lower.scatter_reduce(0, index, pos, reduce="amin")
    ranges = upper - lower
    cell = cell - torch.diag_embed(torch.diagonal(cell, dim1=-2, dim2=-1)) + torch.diag_embed(
        diagonal
    )
    too_small = (diagonal < 2.0 * ranges).any(dim=1)
    if too_small.any():
        cube = 2.0 * ranges.max(dim=1).values
        cell[too_small] = eye * cube[too_small].view(-1, 1, 1)
    return cell


class LocalSplitChargesTorchSimModel(ModelInterface):
    """TorchSim ``ModelInterface`` around a mace_scf ``LocalSplitCharges`` model.

    Args:
        model: Path to a saved model (``torch.save`` of the module) or the module itself.
        device: Device to run on (default: CUDA if available).
        dtype: Floating-point dtype the model is cast to and inputs are expected in.
        pbc_handling: If given, passed to ``model.coulomb_energy.set_pbc_handling``
            (same meaning as the ASE calculator's argument: ``"realspace"`` for clusters,
            anything else for the k-space path). ``None`` keeps the checkpoint's value.
        neighbor_list_fn: A TorchSim batched neighbour-list function with the signature
            of ``torch_sim.neighbors.torchsim_nl`` (default: that one, which picks
            nvalchemiops if installed).
        compute_forces: Return forces.
        compute_stress: Return the stress tensor.
        head: Head name or index (default: the model's first head, as the calculator).
        formal_charges_key: Per-atom state extra read as the formal charges.
        external_field_key: Per-system state extra read as the external field [S, 3].
        return_electrostatics: Also return ``partial_charges`` [N], ``dipole`` [S, 3] and
            ``density_coefficients`` [N, (l+1)^2] (TorchSim stores non-canonical outputs
            in the state extras).
    """

    def __init__(
        self,
        model: Union[str, Path, torch.nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        pbc_handling: Optional[str] = None,
        neighbor_list_fn: Optional[Callable] = None,
        compute_forces: bool = True,
        compute_stress: bool = True,
        head: Optional[Union[str, int]] = None,
        formal_charges_key: str = "formal_charges",
        external_field_key: str = "external_E_field",
        return_electrostatics: bool = False,
    ) -> None:
        if _TORCHSIM_IMPORT_ERROR is not None:
            raise ImportError(
                "LocalSplitChargesTorchSimModel requires torch-sim-atomistic "
                "(pip install torch-sim-atomistic)."
            ) from _TORCHSIM_IMPORT_ERROR
        super().__init__()
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"
        self.neighbor_list_fn = neighbor_list_fn or torchsim_nl
        self.formal_charges_key = formal_charges_key
        self.external_field_key = external_field_key
        self.return_electrostatics = return_electrostatics

        if isinstance(model, (str, Path)):
            model = torch.load(str(model), map_location=self._device, weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a path or a torch.nn.Module")
        if not hasattr(model, "coulomb_energy"):
            raise TypeError(
                f"{type(model).__name__} is not a mace_scf local-source model "
                "(no coulomb_energy); use mace.calculators.mace_torchsim for MACE models"
            )
        self.model = model.to(device=self._device, dtype=self._dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if pbc_handling is not None:
            self.model.coulomb_energy.set_pbc_handling(pbc_handling)

        self.r_max = float(self.model.r_max)
        self.z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.max_l = int(self.model.coulomb_energy.density_max_l)
        heads = list(getattr(self.model, "heads", ["Default"]))
        if head is None:
            self._head_index = 0
        elif isinstance(head, int):
            if not 0 <= head < len(heads):
                raise ValueError(f"head index {head} out of range for {heads}")
            self._head_index = head
        else:
            if head not in heads:
                raise ValueError(f"head {head!r} not in {heads}")
            self._head_index = heads.index(head)

        # Species encoding cache: rebuilt only when the composition/batching changes.
        self._cached_numbers: Optional[torch.Tensor] = None
        self._cached_system_idx: Optional[torch.Tensor] = None
        self._node_attrs: Optional[torch.Tensor] = None
        self._ptr: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    @property
    def pbc_handling(self) -> str:
        return str(self.model.coulomb_energy.pbc_handling)

    def _species_encoding(self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor):
        if (
            self._cached_numbers is None
            or self._cached_numbers.shape != atomic_numbers.shape
            or not torch.equal(self._cached_numbers, atomic_numbers)
            or not torch.equal(self._cached_system_idx, system_idx)
        ):
            indices = atomic_numbers_to_indices(
                atomic_numbers.detach().cpu().numpy(), z_table=self.z_table
            )
            indices = torch.as_tensor(indices, dtype=torch.long, device=self._device)
            node_attrs = torch.zeros(
                (indices.shape[0], len(self.z_table)), dtype=self._dtype, device=self._device
            )
            node_attrs.scatter_(1, indices.view(-1, 1), 1.0)
            counts = torch.bincount(system_idx)
            ptr = torch.zeros(counts.shape[0] + 1, dtype=torch.long, device=self._device)
            ptr[1:] = torch.cumsum(counts, 0)
            self._cached_numbers = atomic_numbers.clone()
            self._cached_system_idx = system_idx.clone()
            self._node_attrs = node_attrs
            self._ptr = ptr
        return self._node_attrs, self._ptr

    def forward(self, state: "ts.SimState", **_kwargs) -> Dict[str, torch.Tensor]:
        if state.device != self._device or state.dtype != self._dtype:
            state = state.to(self._device, self._dtype)
        system_idx = state.system_idx
        n_systems = state.n_systems
        n_atoms = state.n_atoms
        pbc = state.pbc
        periodic = bool(pbc.any())
        node_attrs, ptr = self._species_encoding(state.atomic_numbers, system_idx)

        row_cell = state.row_vector_cell
        if periodic:
            positions = ts.transforms.pbc_wrap_batched(
                state.positions, state.cell, system_idx, pbc
            )
        else:
            positions = state.positions
            row_cell = nonperiodic_cell(row_cell, positions, system_idx)
        positions = positions.detach().clone()

        cutoff = torch.as_tensor(self.r_max, dtype=self._dtype, device=self._device)
        edge_index, edge_system, unit_shifts = self.neighbor_list_fn(
            positions, row_cell, pbc, cutoff, system_idx
        )
        unit_shifts = unit_shifts.to(self._dtype)
        shifts = ts.transforms.compute_cell_shifts(row_cell, unit_shifts, edge_system)

        if state.has_extras(self.formal_charges_key):
            formal_charges = getattr(state, self.formal_charges_key).to(self._dtype)
        else:
            formal_charges = torch.zeros(n_atoms, dtype=self._dtype, device=self._device)
        if state.has_extras(self.external_field_key):
            external_field = getattr(state, self.external_field_key).to(self._dtype)
            external_field = external_field.reshape(n_systems, 3)
        else:
            external_field = torch.zeros(
                (n_systems, 3), dtype=self._dtype, device=self._device
            )

        data: Dict[str, torch.Tensor] = {
            "positions": positions,
            "node_attrs": node_attrs,
            "batch": system_idx,
            "ptr": ptr,
            "head": torch.full(
                (n_systems,), self._head_index, dtype=torch.long, device=self._device
            ),
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": row_cell.reshape(-1, 3),
            "pbc": pbc.view(1, 3).expand(n_systems, 3).reshape(-1),
            "charges": formal_charges,
            "external_field": external_field,
        }
        # graph_longrange's realspace electrostatics allocates its work buffers with
        # torch.get_default_dtype(); under the stock float32 default a float64 model
        # silently loses precision there (1e-5 eV on a 13-water cluster). The ASE
        # calculator sets the global default instead; scope it to the call here.
        previous_default = torch.get_default_dtype()
        torch.set_default_dtype(self._dtype)
        try:
            out = self.model(
                data,
                training=False,
                compute_force=self._compute_forces,
                compute_stress=self._compute_stress,
            )
        finally:
            torch.set_default_dtype(previous_default)

        results: Dict[str, torch.Tensor] = {"energy": out["energy"].detach()}
        if self._compute_forces:
            results["forces"] = out["forces"].detach()
        if self._compute_stress:
            results["stress"] = out["stress"].detach().view(n_systems, 3, 3)
        if self.return_electrostatics:
            density = out["density_coefficients"].detach()
            results["partial_charges"] = density[:, 0]
            results["density_coefficients"] = density
            results["dipole"] = out["dipole"].detach()
        return results
