"""Hessian-projection training: Hutchinson probe sketches of the Hessian error.

Projected Hessian Learning (arXiv:2603.04523) replaces the elementwise Hessian loss
``||H_theta - H_reference||_F^2`` by its unbiased single-probe estimate
``|| (H_theta - H_reference) v ||^2``, which needs only Hessian-vector products (HVPs)
and never forms a Hessian. Two targets are implemented:

``full``
    the plain estimator, unbiased for the mean squared element of the whole Hessian
    error.

``intermolecular``
    the masked bipartition estimator, unbiased for the mean squared element of the
    OFF-block-diagonal (molecule a versus molecule b, a != b) Hessian error. Every
    molecule joins the receiver set or the source set with probability one half, the
    probe is supported on the source molecules only and the response is read out on the
    receiver molecules; each ordered pair survives with probability one quarter, hence
    the factor four. Draws where one side is empty are kept -- they contribute a zero
    estimate, and rejecting them would bias the expectation.

The intermolecular target exists because the intermolecular block is a small-difference
quantity: it holds only a few percent of the squared Hessian error of these water
clusters, so an unmasked Frobenius loss spends almost all of its gradient on covalent
stretch couplings that are already accurate.

Cost. The model-side HVP is ONE backward pass for the whole batch: structures in a batch
are independent graphs, so the per-configuration probes concatenate into a single
batch-wide cotangent and one ``torch.autograd.grad`` call returns every structure's
``H_theta v``. The trainer already builds the forces with ``create_graph=True``
(``mace.modules.utils.compute_forces`` with ``training=True``), so nothing in the model
has to change. The model's own ``compute_hessian`` path is deliberately NOT used: it
needs ``3N`` backward passes over a dense identity and returns with
``create_graph=False``, which makes it both far too slow for a training loop and a dead
end for a loss.

Probe scale is not a free parameter. Unbiasedness requires ``E[v v^T] = I``; scaling the
probes by ``c`` scales every estimate by exactly ``c^2``, which would silently rescale
the loss weight. Both implemented distributions have unit component variance, and any
scaling belongs in the loss weight instead.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace.tools import TensorDict
from mace.tools.scatter import scatter_sum
from mace.tools.torch_geometric import Batch

REFERENCE_HESSIAN_KEY = "reference_hessian_flat"
MOLECULE_ASSIGNMENT_KEY = "molecule_assignment"

PROBE_DISTRIBUTIONS = ("rademacher", "gaussian")
TARGETS = ("full", "intermolecular")

SIDECAR_LAYOUT = "concatenated (3N, 3N) row-major, electronvolt per angstrom squared"
_POSITION_TOLERANCE_ANGSTROM = 1e-5


# --------------------------------------------------------------------------------------
# reference data
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceHessianEntry:
    """One structure's reference Hessian and molecule partition, as stored."""

    label: str
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    hessian_flat: np.ndarray
    """The (3N, 3N) Hessian flattened row-major, in electronvolt per angstrom squared."""
    molecule_assignment: np.ndarray
    """Which molecule every atom belongs to, values ``0 .. molecule_count - 1``."""

    @property
    def atom_count(self) -> int:
        return int(self.atomic_numbers.size)


def load_reference_hessians(path: str) -> List[ReferenceHessianEntry]:
    """Read the sidecar written by the labelling repository's exporter.

    The sidecar is a single ``.npz`` holding per-structure records concatenated along
    their leading axis, in the frame order of the training xyz file, so the entries can
    be zipped onto a dataset positionally and then verified elementwise.
    """
    with np.load(path, allow_pickle=False) as archive:
        layout = str(archive["layout"])
        if layout != SIDECAR_LAYOUT:
            raise ValueError(
                f"reference Hessian sidecar {path} has layout {layout!r}, "
                f"expected {SIDECAR_LAYOUT!r}"
            )
        labels = [str(label) for label in archive["labels"]]
        atom_counts = archive["atom_counts"].astype(int)
        atomic_numbers = archive["atomic_numbers"]
        positions = archive["positions_angstrom"]
        molecule_assignments = archive["molecule_assignments"]
        hessians = archive["hessians_electronvolt_per_angstrom_squared"]

    atom_offsets = np.concatenate([[0], np.cumsum(atom_counts)])
    hessian_lengths = 9 * atom_counts**2
    hessian_offsets = np.concatenate([[0], np.cumsum(hessian_lengths)])
    if atom_offsets[-1] != atomic_numbers.size or hessian_offsets[-1] != hessians.size:
        raise ValueError(f"reference Hessian sidecar {path} is internally inconsistent")

    entries = []
    for index, label in enumerate(labels):
        atom_slice = slice(atom_offsets[index], atom_offsets[index + 1])
        entries.append(
            ReferenceHessianEntry(
                label=label,
                atomic_numbers=atomic_numbers[atom_slice],
                positions_angstrom=positions[atom_slice],
                hessian_flat=hessians[
                    hessian_offsets[index] : hessian_offsets[index + 1]
                ],
                molecule_assignment=molecule_assignments[atom_slice],
            )
        )
    return entries


def attach_reference_hessians(
    dataset: Sequence,
    path: str,
    z_table,
    split_name: str = "dataset",
) -> None:
    """Put a reference Hessian and a molecule partition on every configuration.

    The two fields are set on the already-built ``ExtAtomicData`` objects rather than
    threaded through ``from_config``, because ``torch_geometric``'s collater picks up
    any attribute present on the first item of a batch. ``reference_hessian_flat`` is
    one dimensional, so the default collation (concatenate along dimension zero) is
    exactly right and the per-configuration matrices are recovered from ``ptr``;
    ``molecule_assignment`` is node shaped and behaves like ``forces``. Neither name may
    contain ``index`` or ``face``: the collater matches those substrings and would
    silently concatenate along the last dimension and add the node count.

    The sidecar is aligned positionally with the dataset, so every entry is verified
    against the configuration it lands on -- atom count, atomic numbers and positions.
    A stale or reordered sidecar therefore fails loudly instead of training on Hessians
    that belong to other geometries.
    """
    entries = load_reference_hessians(path)
    if len(entries) != len(dataset):
        raise ValueError(
            f"reference Hessian sidecar {path} holds {len(entries)} structures but the "
            f"{split_name} has {len(dataset)} configurations. The sidecar is aligned "
            "frame by frame with one xyz file: pass the sidecar that belongs to this "
            "split, and use an explicit --valid_file rather than --valid_fraction, "
            "whose split happens inside the loader"
        )

    atomic_numbers_of_index = np.asarray(z_table.zs, dtype=int)
    default_dtype = torch.get_default_dtype()
    for entry, configuration in zip(entries, dataset):
        node_attributes = configuration.node_attrs.numpy()
        atom_count = node_attributes.shape[0]
        if atom_count != entry.atom_count:
            raise ValueError(
                f"reference Hessian for {entry.label} has {entry.atom_count} atoms, "
                f"the {split_name} configuration has {atom_count}"
            )
        configuration_atomic_numbers = atomic_numbers_of_index[
            node_attributes.argmax(axis=1)
        ]
        if not np.array_equal(configuration_atomic_numbers, entry.atomic_numbers):
            raise ValueError(
                f"reference Hessian for {entry.label} does not match the {split_name} "
                "configuration it was zipped onto: atomic numbers differ"
            )
        position_deviation = np.abs(
            configuration.positions.numpy() - entry.positions_angstrom
        ).max()
        if position_deviation > _POSITION_TOLERANCE_ANGSTROM:
            raise ValueError(
                f"reference Hessian for {entry.label} does not match the {split_name} "
                f"configuration it was zipped onto: positions differ by "
                f"{position_deviation:.3e} angstrom"
            )

        setattr(
            configuration,
            REFERENCE_HESSIAN_KEY,
            torch.tensor(entry.hessian_flat, dtype=default_dtype),
        )
        setattr(
            configuration,
            MOLECULE_ASSIGNMENT_KEY,
            torch.tensor(entry.molecule_assignment, dtype=torch.long),
        )

    logging.info(
        f"attached reference Hessians to {len(dataset)} {split_name} configurations "
        f"from {path}"
    )


# --------------------------------------------------------------------------------------
# probes, masks and Hessian-vector products
# --------------------------------------------------------------------------------------


def sample_probe(
    node_count: int,
    device: torch.device,
    dtype: torch.dtype,
    distribution: str = "rademacher",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """A probe with ``E[v v^T] = I``, shaped like the positions: ``(node_count, 3)``."""
    if distribution == "rademacher":
        signs = torch.randint(
            0, 2, (node_count, 3), device=device, generator=generator
        )
        return 2.0 * signs.to(dtype) - 1.0
    if distribution == "gaussian":
        return torch.randn(
            (node_count, 3), device=device, dtype=dtype, generator=generator
        )
    raise ValueError(
        f"unknown probe distribution {distribution!r}, expected one of "
        f"{PROBE_DISTRIBUTIONS}"
    )


def global_molecule_indices(batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch-wide molecule labels from the per-configuration molecule assignment.

    Returns the molecule label of every node and the graph label of every molecule.
    """
    # Imported here, not at module scope: mace_scf.electrostatics imports the loss
    # registry -- which imports this module -- before its own utils would be reachable.
    from mace_scf.electrostatics.utils import compute_effective_index

    if not hasattr(batch, MOLECULE_ASSIGNMENT_KEY):
        raise ValueError(
            "the batch carries no molecule assignment; call attach_reference_hessians "
            "on the dataset before training with an intermolecular Hessian loss"
        )
    molecule_of_node, combinations = compute_effective_index(
        [batch["batch"], batch[MOLECULE_ASSIGNMENT_KEY]]
    )
    return molecule_of_node, combinations[:, 0]


def intermolecular_element_counts(batch: Batch) -> torch.Tensor:
    """Per configuration, the number of intermolecular Hessian elements.

    ``9 * sum_{a != b} N_a N_b = 9 * (N^2 - sum_a N_a^2)``; zero for a single molecule.
    """
    molecule_of_node, graph_of_molecule = global_molecule_indices(batch)
    graph_count = batch.ptr.numel() - 1
    molecule_count = int(graph_of_molecule.numel())
    atoms_per_molecule = scatter_sum(
        src=torch.ones_like(molecule_of_node),
        index=molecule_of_node,
        dim=0,
        dim_size=molecule_count,
    )
    atoms_per_graph = batch.ptr[1:] - batch.ptr[:-1]
    squared_atoms_per_graph = scatter_sum(
        src=atoms_per_molecule**2,
        index=graph_of_molecule,
        dim=0,
        dim_size=graph_count,
    )
    return 9 * (atoms_per_graph**2 - squared_atoms_per_graph)


def molecule_bipartition(
    batch: Batch, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split the molecules of every configuration into receivers and sources.

    Every molecule joins the receiver side independently with probability one half, so
    an ordered molecule pair is retained with probability one quarter. Returns the
    per-node receiver and source masks.
    """
    molecule_of_node, graph_of_molecule = global_molecule_indices(batch)
    membership = torch.randint(
        0,
        2,
        (int(graph_of_molecule.numel()),),
        device=molecule_of_node.device,
        generator=generator,
    ).bool()
    receiver_mask = membership[molecule_of_node]
    return receiver_mask, ~receiver_mask


def hessian_vector_product(
    forces: torch.Tensor,
    positions: torch.Tensor,
    probe: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """``H_theta v`` for every configuration of the batch in ONE backward pass.

    ``H v = d/dx (v . dE/dx) = -d/dx (v . F)``, so a single vector-Jacobian product of
    the forces with the probe as cotangent gives the Hessian-vector product. The batch
    Hessian is block diagonal in the configurations, so the concatenated probe yields
    every configuration's product at once. ``create_graph`` must be true in a training
    loss so the result can be differentiated with respect to the model parameters.
    """
    product = torch.autograd.grad(
        outputs=[-1 * forces.reshape(-1)],
        inputs=[positions],
        grad_outputs=[probe.reshape(-1)],
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=False,
    )[0]
    if product is None:
        raise ValueError(
            "the Hessian-vector product is unused: the forces do not depend on the "
            "positions tensor that was differentiated"
        )
    return product.view_as(positions)


def reference_hessian_vector_product(batch: Batch, probe: torch.Tensor) -> torch.Tensor:
    """``H_reference v`` per configuration, from the stored Hessians on the batch."""
    if not hasattr(batch, REFERENCE_HESSIAN_KEY):
        raise ValueError(
            "the batch carries no reference Hessians; call attach_reference_hessians "
            "on the dataset before training with a Hessian loss"
        )
    stored = batch[REFERENCE_HESSIAN_KEY]
    node_offsets = batch.ptr
    atoms_per_graph = (node_offsets[1:] - node_offsets[:-1]).tolist()

    responses = []
    hessian_offset = 0
    for graph_index, atom_count in enumerate(atoms_per_graph):
        coordinate_count = 3 * atom_count
        hessian_length = coordinate_count**2
        hessian = stored[
            hessian_offset : hessian_offset + hessian_length
        ].view(coordinate_count, coordinate_count)
        hessian_offset += hessian_length
        graph_probe = probe[
            node_offsets[graph_index] : node_offsets[graph_index + 1]
        ].reshape(-1)
        responses.append((hessian @ graph_probe.to(hessian.dtype)).view(atom_count, 3))
    if hessian_offset != stored.numel():
        raise ValueError(
            "the stored reference Hessians do not match the batch's atom counts"
        )
    return torch.cat(responses, dim=0).to(probe.dtype)


# --------------------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionDraw:
    """One realisation of the estimator's randomness, before anything is evaluated.

    Kept separate from the arithmetic so that the unbiasedness of the estimator can be
    pinned by enumerating every draw exactly, rather than sampled to a tolerance.
    """

    probe: torch.Tensor
    """The probe, already restricted to the source side; shaped like the positions."""
    readout_mask: Optional[torch.Tensor]
    """Which coordinates the response is read out on, or ``None`` for all of them."""
    prefactor: float
    element_count: torch.Tensor


@dataclass(frozen=True)
class ProjectionSample:
    """One draw's estimates, per configuration of the batch.

    ``error_squared`` is unbiased for ``||dH_block||_F^2`` and ``reference_squared`` for
    ``||H_reference,block||_F^2``, both including the estimator's prefactor;
    ``element_count`` is the number of Hessian elements of the targeted block, so
    ``error_squared / element_count`` is a mean squared element.
    """

    error_squared: torch.Tensor
    reference_squared: torch.Tensor
    element_count: torch.Tensor


def draw_projection(
    batch: Batch,
    target: str,
    node_count: int,
    device: torch.device,
    dtype: torch.dtype,
    distribution: str = "rademacher",
    generator: Optional[torch.Generator] = None,
) -> ProjectionDraw:
    """Draw one probe (and, for the intermolecular target, one bipartition)."""
    if target not in TARGETS:
        raise ValueError(f"unknown Hessian target {target!r}, expected one of {TARGETS}")

    probe = sample_probe(
        node_count=node_count,
        device=device,
        dtype=dtype,
        distribution=distribution,
        generator=generator,
    )
    if target == "full":
        atoms_per_graph = batch.ptr[1:] - batch.ptr[:-1]
        return ProjectionDraw(
            probe=probe,
            readout_mask=None,
            prefactor=1.0,
            element_count=9 * atoms_per_graph**2,
        )

    receiver_mask, source_mask = molecule_bipartition(batch, generator=generator)
    return ProjectionDraw(
        probe=probe * source_mask.unsqueeze(-1),
        readout_mask=receiver_mask.unsqueeze(-1),
        prefactor=4.0,
        element_count=intermolecular_element_counts(batch),
    )


def projection_estimates(
    batch: Batch,
    forces: torch.Tensor,
    positions: torch.Tensor,
    draw: ProjectionDraw,
    create_graph: bool,
) -> ProjectionSample:
    """Evaluate one draw: one batch-wide HVP, one reference matvec, then reduce."""
    graph_count = batch.ptr.numel() - 1
    model_response = hessian_vector_product(
        forces=forces, positions=positions, probe=draw.probe, create_graph=create_graph
    )
    reference_response = reference_hessian_vector_product(batch, draw.probe)
    residual = model_response - reference_response
    if draw.readout_mask is not None:
        residual = residual * draw.readout_mask
        reference_response = reference_response * draw.readout_mask

    return ProjectionSample(
        error_squared=draw.prefactor
        * scatter_sum(
            src=(residual**2).sum(dim=-1),
            index=batch["batch"],
            dim=0,
            dim_size=graph_count,
        ),
        reference_squared=draw.prefactor
        * scatter_sum(
            src=(reference_response**2).sum(dim=-1),
            index=batch["batch"],
            dim=0,
            dim_size=graph_count,
        ),
        element_count=draw.element_count,
    )


def projection_sample(
    batch: Batch,
    forces: torch.Tensor,
    positions: torch.Tensor,
    target: str,
    create_graph: bool,
    distribution: str = "rademacher",
    generator: Optional[torch.Generator] = None,
) -> ProjectionSample:
    """Draw one probe and return its per-configuration estimates."""
    draw = draw_projection(
        batch=batch,
        target=target,
        node_count=positions.shape[0],
        device=positions.device,
        dtype=positions.dtype,
        distribution=distribution,
        generator=generator,
    )
    return projection_estimates(
        batch=batch,
        forces=forces,
        positions=positions,
        draw=draw,
        create_graph=create_graph,
    )


class HessianProjectionLoss(torch.nn.Module):
    """Mean squared Hessian element of the targeted block, Hutchinson estimated.

    Registered as the ``hessian_projection`` loss term, so a training stage declares it
    as, for example::

        loss:
          energy_per_atom: 10.0
          forces: 100.0
          hessian_projection:
            weight: 100.0
            target: intermolecular
            probe_count: 1

    One fresh probe per configuration per batch is the intended operating point: a
    single Rademacher probe estimates the loss of a mid-sized cluster to roughly twenty
    percent, which is far below the gradient noise stochastic gradient descent already
    tolerates, and the estimator is unbiased at every batch.

    The term is a training-time term. The trainer's validation pass detaches the model
    outputs before calling the loss, so no second derivative is available there and the
    term contributes zero; the validation counterpart is
    :func:`evaluate_hessian_projections`.
    """

    def __init__(
        self,
        target: str = "intermolecular",
        probe_count: int = 1,
        probe_distribution: str = "rademacher",
    ):
        super().__init__()
        if target not in TARGETS:
            raise ValueError(f"unknown Hessian target {target!r}, expected one of {TARGETS}")
        if probe_distribution not in PROBE_DISTRIBUTIONS:
            raise ValueError(
                f"unknown probe distribution {probe_distribution!r}, expected one of "
                f"{PROBE_DISTRIBUTIONS}"
            )
        if probe_count < 1:
            raise ValueError("probe_count must be at least one")
        self.target = target
        self.probe_count = probe_count
        self.probe_distribution = probe_distribution

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        forces = pred.get("forces")
        if forces is None:
            raise ValueError(
                "the Hessian projection loss needs forces; enable compute_forces"
            )
        positions = pred.get("positions")
        if positions is None:
            raise ValueError(
                "the Hessian projection loss needs the positions tensor the forward "
                "differentiated, which the model wrapper puts into the output as "
                "'positions'. This model's wrapper does not; the fixed-point models in "
                "particular have no Hessian path at all (a second derivative through "
                "the self-consistency loop needs implicit differentiation)."
            )
        if not forces.requires_grad:
            # Validation path: the trainer detaches the outputs before calling the loss.
            return torch.zeros((), device=forces.device, dtype=forces.dtype)

        graph_count = ref.ptr.numel() - 1
        total = torch.zeros((), device=forces.device, dtype=forces.dtype)
        for _ in range(self.probe_count):
            sample = projection_sample(
                batch=ref,
                forces=forces,
                positions=positions,
                target=self.target,
                create_graph=True,
                distribution=self.probe_distribution,
            )
            # Configurations without a targeted block -- a monomer has no
            # intermolecular elements -- contribute nothing and are not averaged over.
            contributes = sample.element_count > 0
            element_count = sample.element_count.clamp(min=1).to(total.dtype)
            per_configuration = (
                ref.weight * sample.error_squared / element_count
            ) * contributes.to(total.dtype)
            total = total + per_configuration.sum() / contributes.sum().clamp(min=1)
        return total / self.probe_count

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(target={self.target}, "
            f"probe_count={self.probe_count}, "
            f"probe_distribution={self.probe_distribution})"
        )


# --------------------------------------------------------------------------------------
# validation metric
# --------------------------------------------------------------------------------------


def evaluate_hessian_projections(
    model: torch.nn.Module,
    model_eval_wrapper,
    data_loader,
    device: torch.device,
    probe_count: int = 16,
    seed: int = 0,
    ema=None,
) -> Dict[str, float]:
    """Pooled Hessian errors on a data loader, for BOTH targets.

    Both targets are reported whichever one is being trained -- that cross comparison is
    the point of the study. Probes come from a fixed seed so the numbers are comparable
    across epochs, arms and runs, and are averaged over ``probe_count`` draws to keep the
    estimator noise well below the model differences being measured.

    Reported per target: ``rmse_hessian_<target>``, the root mean squared Hessian element
    error in electronvolt per angstrom squared, and ``rel_hessian_<target>``, that error
    relative to the reference curvature of the same block -- the same pooled definition
    the offline block benchmark uses, so the two can be compared directly.

    Unlike the trainer's own validation pass this one runs the forward with
    ``training=True``, because the second derivative needs the force graph. The
    exponential moving average is therefore entered by hand: the wrapper only applies it
    when ``training`` is false.
    """
    error_squared = {target: 0.0 for target in TARGETS}
    reference_squared = {target: 0.0 for target in TARGETS}
    element_count = {target: 0.0 for target in TARGETS}

    parameter_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    average_parameters = (
        ema.average_parameters() if ema is not None else nullcontext()
    )
    with average_parameters:
        for batch_number, batch in enumerate(data_loader):
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model_eval_wrapper(model, batch_dict, training=True)
            positions = output.get("positions")
            if positions is None:
                raise ValueError(
                    "the Hessian projection evaluation needs the positions tensor the "
                    "forward differentiated in the model output"
                )
            generator = torch.Generator(device=positions.device)
            for target in TARGETS:
                for probe_number in range(probe_count):
                    generator.manual_seed(
                        seed
                        + 1_000_003 * batch_number
                        + 1_009 * probe_number
                        + TARGETS.index(target)
                    )
                    sample = projection_sample(
                        batch=batch,
                        forces=output["forces"],
                        positions=positions,
                        target=target,
                        create_graph=False,
                        generator=generator,
                    )
                    contributes = (sample.element_count > 0).to(sample.error_squared.dtype)
                    error_squared[target] += float(
                        (sample.error_squared.detach() * contributes).sum()
                    )
                    reference_squared[target] += float(
                        (sample.reference_squared.detach() * contributes).sum()
                    )
                    element_count[target] += float(
                        (sample.element_count.to(contributes.dtype) * contributes).sum()
                    )
            del output

    for name, parameter in model.named_parameters():
        if name in parameter_requires_grad:
            parameter.requires_grad_(parameter_requires_grad[name])
        parameter.grad = None

    metrics: Dict[str, float] = {}
    for target in TARGETS:
        if element_count[target] == 0.0:
            continue
        metrics[f"rmse_hessian_{target}"] = float(
            np.sqrt(error_squared[target] / element_count[target])
        )
        if reference_squared[target] > 0.0:
            metrics[f"rel_hessian_{target}"] = float(
                np.sqrt(error_squared[target] / reference_squared[target])
            )
    return metrics


def build_hessian_projection_evaluation(
    probe_count: int = 16, seed: int = 0
) -> Callable[..., Dict[str, float]]:
    """A ``train``-compatible validation hook with the probe budget bound in."""

    def evaluation(model, model_eval_wrapper, data_loader, device, ema=None):
        return evaluate_hessian_projections(
            model=model,
            model_eval_wrapper=model_eval_wrapper,
            data_loader=data_loader,
            device=device,
            probe_count=probe_count,
            seed=seed,
            ema=ema,
        )

    return evaluation
