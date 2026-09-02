"""The Hessian-projection estimators: exactness, unbiasedness and the guards.

The model is replaced by a surrogate whose Hessian is a matrix chosen up front --
``forces = -(A x)`` has ``H = A`` exactly -- so every estimate can be compared against
the quantity it is supposed to estimate. Unbiasedness of both estimators is pinned by
ENUMERATING every draw on a small system (all sign vectors, and for the intermolecular
estimator all bipartitions as well), which fixes the factor four exactly rather than
statistically; a water-sized batch is then checked by sampling.
"""

import itertools

import numpy as np
import pytest
import torch
from ase import Atoms

from mace.tools import torch_geometric

import mace_scf.hessian_projections as hessian_projections
from mace_scf.hessian_projections import (
    HessianProjectionLoss,
    ProjectionDraw,
    attach_reference_hessians,
    draw_projection,
    hessian_vector_product,
    intermolecular_element_counts,
    projection_estimates,
    projection_sample,
    sample_probe,
)
from tests.utils import dataset_from_atoms

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _atoms(atom_count: int) -> Atoms:
    """A chain of atoms, geometry irrelevant: the surrogate ignores it."""
    symbols = "O" + "H" * (atom_count - 1)
    positions = [[1.0 * index, 0.1 * index, 0.0] for index in range(atom_count)]
    return Atoms(symbols=symbols, positions=positions)


def _symmetric(coordinate_count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(coordinate_count, coordinate_count))
    return 0.5 * (matrix + matrix.T)


class _Case:
    """A batch, its surrogate Hessians and its reference Hessians, all consistent."""

    def __init__(self, atom_counts, assignments, seed=1):
        dataset = dataset_from_atoms(
            [_atoms(count) for count in atom_counts], cutoff=3.0
        )
        self.atom_counts = list(atom_counts)
        self.assignments = [np.asarray(assignment) for assignment in assignments]
        self.model_hessians = []
        self.reference_hessians = []
        for index, (count, assignment) in enumerate(zip(atom_counts, assignments)):
            self.model_hessians.append(_symmetric(3 * count, seed + 10 * index))
            self.reference_hessians.append(_symmetric(3 * count, seed + 10 * index + 5))
            dataset[index].reference_hessian_flat = torch.tensor(
                self.reference_hessians[-1].reshape(-1)
            )
            dataset[index].molecule_assignment = torch.tensor(
                np.asarray(assignment), dtype=torch.long
            )
        self.batch = torch_geometric.Batch.from_data_list(dataset)
        self.positions = self.batch.positions.clone().requires_grad_(True)
        block_diagonal = torch.zeros(
            (3 * sum(atom_counts), 3 * sum(atom_counts)), dtype=torch.float64
        )
        offset = 0
        for count, hessian in zip(atom_counts, self.model_hessians):
            block_diagonal[
                offset : offset + 3 * count, offset : offset + 3 * count
            ] = torch.tensor(hessian)
            offset += 3 * count
        self.model_hessian_batch = block_diagonal
        self.forces = -(block_diagonal @ self.positions.reshape(-1)).view(-1, 3)

    @property
    def errors(self):
        return [
            model - reference
            for model, reference in zip(self.model_hessians, self.reference_hessians)
        ]

    def estimate(self, draw):
        return projection_estimates(
            batch=self.batch,
            forces=self.forces,
            positions=self.positions,
            draw=draw,
            create_graph=False,
        )

    def predictions(self):
        return {"forces": self.forces, "positions": self.positions}


def _exact_full_squared(error: np.ndarray) -> float:
    return float((error**2).sum())


def _exact_intermolecular_squared(error: np.ndarray, assignment: np.ndarray) -> float:
    coordinate_assignment = np.repeat(assignment, 3)
    total = float((error**2).sum())
    for molecule in np.unique(assignment):
        mask = coordinate_assignment == molecule
        total -= float((error[np.ix_(mask, mask)] ** 2).sum())
    return total


def _all_sign_vectors(dimension: int) -> np.ndarray:
    return np.array(list(itertools.product([-1.0, 1.0], repeat=dimension)))


def _write_sidecar(path, labels, atomic_numbers, positions, assignments, hessians):
    np.savez(
        path,
        layout=np.array(hessian_projections.SIDECAR_LAYOUT),
        labels=np.array(labels),
        atom_counts=np.array(
            [numbers.size for numbers in atomic_numbers], dtype=np.int64
        ),
        atomic_numbers=np.concatenate(atomic_numbers).astype(np.int64),
        positions_angstrom=np.concatenate(positions).astype(np.float64),
        molecule_assignments=np.concatenate(assignments).astype(np.int64),
        hessians_electronvolt_per_angstrom_squared=np.concatenate(
            [hessian.reshape(-1) for hessian in hessians]
        ).astype(np.float64),
    )


# --------------------------------------------------------------------------------------
# the Hessian-vector product
# --------------------------------------------------------------------------------------


def test_hessian_vector_product_reproduces_the_known_hessian():
    case = _Case(atom_counts=[3, 4], assignments=[[0, 0, 1], [0, 0, 1, 1]])
    probe = sample_probe(
        node_count=case.positions.shape[0], device=torch.device("cpu"), dtype=torch.float64
    )
    product = hessian_vector_product(
        forces=case.forces, positions=case.positions, probe=probe, create_graph=False
    )
    expected = (case.model_hessian_batch @ probe.reshape(-1)).view(-1, 3)
    torch.testing.assert_close(product, expected, rtol=0.0, atol=1e-12)


def test_one_backward_pass_serves_every_configuration_of_the_batch():
    """The batch Hessian is block diagonal, so one product covers all configurations."""
    case = _Case(atom_counts=[2, 3, 4], assignments=[[0, 1], [0, 0, 1], [0, 1, 1, 0]])
    probe = sample_probe(
        node_count=case.positions.shape[0], device=torch.device("cpu"), dtype=torch.float64
    )
    product = hessian_vector_product(
        forces=case.forces, positions=case.positions, probe=probe, create_graph=False
    )
    offset = 0
    for count, hessian in zip(case.atom_counts, case.model_hessians):
        graph_probe = probe[offset : offset + count].reshape(-1).numpy()
        torch.testing.assert_close(
            product[offset : offset + count].reshape(-1),
            torch.tensor(hessian @ graph_probe),
            rtol=0.0,
            atol=1e-12,
        )
        offset += count


# --------------------------------------------------------------------------------------
# unbiasedness, by enumeration
# --------------------------------------------------------------------------------------


def test_full_estimator_is_exact_averaged_over_all_sign_vectors():
    case = _Case(atom_counts=[2], assignments=[[0, 1]])
    element_count = 9 * (case.batch.ptr[1:] - case.batch.ptr[:-1]) ** 2

    probes = _all_sign_vectors(6)
    total = 0.0
    for probe in probes:
        draw = ProjectionDraw(
            probe=torch.tensor(probe).view(2, 3),
            readout_mask=None,
            prefactor=1.0,
            element_count=element_count,
        )
        total += float(case.estimate(draw).error_squared[0])

    assert total / len(probes) == pytest.approx(
        _exact_full_squared(case.errors[0]), rel=1e-12
    )


def test_intermolecular_estimator_is_exact_averaged_over_probes_and_bipartitions():
    """Pins the factor four: every ordered molecule pair survives with probability 1/4."""
    case = _Case(atom_counts=[2], assignments=[[0, 1]])
    element_count = intermolecular_element_counts(case.batch)
    assert element_count.tolist() == [18]

    probes = _all_sign_vectors(6)
    memberships = list(itertools.product([False, True], repeat=2))
    total = 0.0
    for membership in memberships:
        receiver = torch.tensor(membership).view(2, 1)
        for probe in probes:
            draw = ProjectionDraw(
                probe=torch.tensor(probe).view(2, 3) * ~receiver,
                readout_mask=receiver,
                prefactor=4.0,
                element_count=element_count,
            )
            total += float(case.estimate(draw).error_squared[0])

    assert total / (len(probes) * len(memberships)) == pytest.approx(
        _exact_intermolecular_squared(case.errors[0], case.assignments[0]), rel=1e-12
    )


def test_reference_estimate_is_exact_averaged_over_all_sign_vectors():
    """The relative validation metric divides by this, so it must be unbiased too."""
    case = _Case(atom_counts=[2], assignments=[[0, 1]])
    element_count = intermolecular_element_counts(case.batch)
    probes = _all_sign_vectors(6)
    memberships = list(itertools.product([False, True], repeat=2))
    total = 0.0
    for membership in memberships:
        receiver = torch.tensor(membership).view(2, 1)
        for probe in probes:
            draw = ProjectionDraw(
                probe=torch.tensor(probe).view(2, 3) * ~receiver,
                readout_mask=receiver,
                prefactor=4.0,
                element_count=element_count,
            )
            total += float(case.estimate(draw).reference_squared[0])

    assert total / (len(probes) * len(memberships)) == pytest.approx(
        _exact_intermolecular_squared(
            case.reference_hessians[0], case.assignments[0]
        ),
        rel=1e-12,
    )


@pytest.mark.parametrize("target", ["full", "intermolecular"])
def test_sampled_estimator_is_unbiased_on_a_water_sized_batch(target):
    case = _Case(atom_counts=[9, 12], assignments=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3])
    exact = [
        _exact_full_squared(error)
        if target == "full"
        else _exact_intermolecular_squared(error, assignment)
        for error, assignment in zip(case.errors, case.assignments)
    ]

    generator = torch.Generator()
    generator.manual_seed(2026)
    draw_count = 4000
    total = torch.zeros(2, dtype=torch.float64)
    for _ in range(draw_count):
        sample = projection_sample(
            batch=case.batch,
            forces=case.forces,
            positions=case.positions,
            target=target,
            create_graph=False,
            generator=generator,
        )
        total += sample.error_squared.detach()
    estimate = (total / draw_count).tolist()

    for value, expected in zip(estimate, exact):
        assert value == pytest.approx(expected, rel=0.06)


@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
def test_probes_have_identity_covariance(distribution):
    """``E[v v^T] = I`` is what makes the estimators unbiased; the scale is not free."""
    generator = torch.Generator()
    generator.manual_seed(7)
    probes = torch.stack(
        [
            sample_probe(
                node_count=2,
                device=torch.device("cpu"),
                dtype=torch.float64,
                distribution=distribution,
                generator=generator,
            ).reshape(-1)
            for _ in range(40000)
        ]
    )
    covariance = (probes.T @ probes) / probes.shape[0]
    torch.testing.assert_close(
        covariance, torch.eye(6, dtype=torch.float64), rtol=0.0, atol=0.03
    )


# --------------------------------------------------------------------------------------
# the loss term
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["full", "intermolecular"])
def test_loss_averages_the_per_element_error_over_configurations(target):
    case = _Case(atom_counts=[3, 4], assignments=[[0, 0, 1], [0, 0, 1, 1]])
    loss = HessianProjectionLoss(target=target)
    generator = torch.Generator()
    generator.manual_seed(11)

    element_counts = (
        (9 * (case.batch.ptr[1:] - case.batch.ptr[:-1]) ** 2)
        if target == "full"
        else intermolecular_element_counts(case.batch)
    )
    exact = [
        (
            _exact_full_squared(error)
            if target == "full"
            else _exact_intermolecular_squared(error, assignment)
        )
        / float(count)
        for error, assignment, count in zip(
            case.errors, case.assignments, element_counts
        )
    ]

    draw_count = 3000
    total = 0.0
    for _ in range(draw_count):
        draw = draw_projection(
            batch=case.batch,
            target=target,
            node_count=case.positions.shape[0],
            device=case.positions.device,
            dtype=case.positions.dtype,
            generator=generator,
        )
        sample = case.estimate(draw)
        contributes = (sample.element_count > 0).double()
        total += float(
            (
                case.batch.weight
                * sample.error_squared.detach()
                / sample.element_count.clamp(min=1).double()
                * contributes
            ).sum()
            / contributes.sum()
        )
    assert total / draw_count == pytest.approx(float(np.mean(exact)), rel=0.08)
    assert isinstance(loss(ref=case.batch, pred=case.predictions()), torch.Tensor)


def test_monomer_contributes_nothing_to_the_intermolecular_loss():
    """A single molecule has no intermolecular block: no division by zero, no NaN."""
    case = _Case(atom_counts=[3], assignments=[[0, 0, 0]])
    assert intermolecular_element_counts(case.batch).tolist() == [0]
    value = HessianProjectionLoss(target="intermolecular")(
        ref=case.batch, pred=case.predictions()
    )
    assert torch.isfinite(value)
    assert float(value) == 0.0


def test_a_monomer_in_a_batch_does_not_dilute_the_intermolecular_loss():
    case = _Case(atom_counts=[3, 4], assignments=[[0, 0, 0], [0, 0, 1, 1]])
    generator = torch.Generator()
    generator.manual_seed(3)
    exact = _exact_intermolecular_squared(
        case.errors[1], case.assignments[1]
    ) / float(intermolecular_element_counts(case.batch)[1])

    draw_count = 3000
    total = 0.0
    for _ in range(draw_count):
        draw = draw_projection(
            batch=case.batch,
            target="intermolecular",
            node_count=case.positions.shape[0],
            device=case.positions.device,
            dtype=case.positions.dtype,
            generator=generator,
        )
        sample = case.estimate(draw)
        contributes = (sample.element_count > 0).double()
        total += float(
            (
                sample.error_squared.detach()
                / sample.element_count.clamp(min=1).double()
                * contributes
            ).sum()
            / contributes.sum()
        )
    # Averaged over the ONE contributing configuration, not over both.
    assert total / draw_count == pytest.approx(exact, rel=0.08)


def test_loss_contributes_zero_without_a_force_graph():
    """The trainer's validation pass detaches the outputs before calling the loss."""
    case = _Case(atom_counts=[3, 4], assignments=[[0, 0, 1], [0, 0, 1, 1]])
    detached = {
        "forces": case.forces.detach(),
        "positions": case.positions.detach(),
    }
    value = HessianProjectionLoss(target="intermolecular")(
        ref=case.batch, pred=detached
    )
    assert float(value) == 0.0


def test_loss_without_the_differentiated_positions_is_rejected():
    case = _Case(atom_counts=[3], assignments=[[0, 0, 1]])
    with pytest.raises(ValueError, match="positions tensor the forward"):
        HessianProjectionLoss(target="full")(
            ref=case.batch, pred={"forces": case.forces}
        )


def test_loss_options_are_validated():
    with pytest.raises(ValueError, match="unknown Hessian target"):
        HessianProjectionLoss(target="off_diagonal")
    with pytest.raises(ValueError, match="unknown probe distribution"):
        HessianProjectionLoss(probe_distribution="one_hot")
    with pytest.raises(ValueError, match="at least one"):
        HessianProjectionLoss(probe_count=0)


def test_loss_is_registered_for_the_weighted_loss():
    from mace_scf.electrostatics.loss import WeightedLoss

    weighted = WeightedLoss(
        {
            "forces": {"weight": 100.0},
            "hessian_projection": {
                "weight": 10.0,
                "target": "intermolecular",
                "probe_count": 2,
            },
        }
    )
    term = weighted.loss_fns["hessian_projection"]
    assert term.target == "intermolecular"
    assert term.probe_count == 2
    assert weighted.loss_weights["hessian_projection"] == 10.0


def test_missing_reference_hessians_are_reported():
    dataset = dataset_from_atoms([_atoms(3)], cutoff=3.0)
    batch = torch_geometric.Batch.from_data_list(dataset)
    positions = batch.positions.clone().requires_grad_(True)
    forces = -(positions * 2.0)
    with pytest.raises(ValueError, match="no reference Hessians"):
        HessianProjectionLoss(target="full")(
            ref=batch, pred={"forces": forces, "positions": positions}
        )


# --------------------------------------------------------------------------------------
# attaching the sidecar
# --------------------------------------------------------------------------------------


def test_attach_reference_hessians_round_trips(tmp_path):
    atoms_list = [_atoms(3), _atoms(4)]
    dataset = dataset_from_atoms(atoms_list, cutoff=3.0)
    z_table = _z_table(atoms_list)
    hessians = [_symmetric(9, 1), _symmetric(12, 2)]
    assignments = [np.array([0, 0, 1]), np.array([0, 0, 1, 1])]
    path = tmp_path / "reference.npz"
    _write_sidecar(
        path,
        labels=["first", "second"],
        atomic_numbers=[atoms.get_atomic_numbers() for atoms in atoms_list],
        positions=[atoms.get_positions() for atoms in atoms_list],
        assignments=assignments,
        hessians=hessians,
    )

    attach_reference_hessians(dataset, str(path), z_table=z_table)
    batch = torch_geometric.Batch.from_data_list(dataset)
    assert batch.reference_hessian_flat.numel() == 81 + 144
    np.testing.assert_array_equal(
        batch.molecule_assignment.numpy(), np.concatenate(assignments)
    )
    np.testing.assert_allclose(
        batch.reference_hessian_flat[:81].numpy().reshape(9, 9), hessians[0]
    )


def test_attach_reference_hessians_rejects_a_shifted_geometry(tmp_path):
    atoms_list = [_atoms(3)]
    dataset = dataset_from_atoms(atoms_list, cutoff=3.0)
    positions = atoms_list[0].get_positions()
    positions[1, 0] += 0.01
    path = tmp_path / "reference.npz"
    _write_sidecar(
        path,
        labels=["first"],
        atomic_numbers=[atoms_list[0].get_atomic_numbers()],
        positions=[positions],
        assignments=[np.array([0, 0, 1])],
        hessians=[_symmetric(9, 1)],
    )
    with pytest.raises(ValueError, match="positions differ"):
        attach_reference_hessians(dataset, str(path), z_table=_z_table(atoms_list))


def test_attach_reference_hessians_rejects_a_different_species(tmp_path):
    atoms_list = [_atoms(3)]
    dataset = dataset_from_atoms(atoms_list, cutoff=3.0)
    path = tmp_path / "reference.npz"
    _write_sidecar(
        path,
        labels=["first"],
        atomic_numbers=[np.array([1, 8, 1])],
        positions=[atoms_list[0].get_positions()],
        assignments=[np.array([0, 0, 1])],
        hessians=[_symmetric(9, 1)],
    )
    with pytest.raises(ValueError, match="atomic numbers differ"):
        attach_reference_hessians(dataset, str(path), z_table=_z_table(atoms_list))


def test_attach_reference_hessians_rejects_a_length_mismatch(tmp_path):
    atoms_list = [_atoms(3), _atoms(4)]
    dataset = dataset_from_atoms(atoms_list, cutoff=3.0)
    path = tmp_path / "reference.npz"
    _write_sidecar(
        path,
        labels=["first"],
        atomic_numbers=[atoms_list[0].get_atomic_numbers()],
        positions=[atoms_list[0].get_positions()],
        assignments=[np.array([0, 0, 1])],
        hessians=[_symmetric(9, 1)],
    )
    with pytest.raises(ValueError, match="holds 1 structures"):
        attach_reference_hessians(dataset, str(path), z_table=_z_table(atoms_list))


def _z_table(atoms_list):
    import mace.tools

    return mace.tools.get_atomic_number_table_from_zs(
        sorted({int(number) for atoms in atoms_list for number in atoms.get_atomic_numbers()})
    )
