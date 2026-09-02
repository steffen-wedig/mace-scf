"""The Hessian-projection loss against a real LocalSplitCharges model.

Two things can only be checked with the actual model in the loop: that the single
batch-wide vector-Jacobian product agrees with the model's own analytic Hessian, and
that the loss is differentiable with respect to the parameters through the second
derivative (a double backward through the e3nn-scripted modules).
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

import mace.modules
import mace.tools
from mace.tools import torch_geometric

import mace_scf.electrostatics
import mace_scf.utils
from mace_scf.hessian_projections import (
    HessianProjectionLoss,
    ProjectionDraw,
    hessian_vector_product,
    intermolecular_element_counts,
    projection_estimates,
    sample_probe,
)
from tests.utils import dataset_from_atoms, disable_e3nn_codegen, seed_torch

torch.set_default_dtype(torch.float64)

CUTOFF = 4.0


def _water_dimer() -> Atoms:
    return Atoms(
        symbols="OHHOHH",
        positions=[
            [0.00, 0.00, 0.00],
            [0.96, 0.00, 0.00],
            [-0.24, 0.93, 0.00],
            [2.80, 0.10, 0.05],
            [3.30, 0.85, 0.20],
            [3.20, -0.60, 0.45],
        ],
    )


def _molecule_assignment(atoms: Atoms) -> np.ndarray:
    """Each hydrogen to its nearest oxygen -- the partition the exporter writes."""
    positions = atoms.get_positions()
    oxygens = [
        index for index, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "O"
    ]
    assignment = np.zeros(len(atoms), dtype=int)
    for index, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == "O":
            assignment[index] = oxygens.index(index)
        else:
            distances = [
                np.linalg.norm(positions[index] - positions[oxygen]) for oxygen in oxygens
            ]
            assignment[index] = int(np.argmin(distances))
    return assignment


def _local_split_charges_model(atoms: Atoms, seed: int = 5):
    seed_torch(seed)
    z_table = mace.tools.get_atomic_number_table_from_zs(
        sorted(set(int(number) for number in atoms.get_atomic_numbers()))
    )
    formal_charges = np.array(
        [1.0 if z == 1 else -2.0 for z in z_table.zs], dtype=float
    )
    with disable_e3nn_codegen():
        model = mace_scf.electrostatics.LocalSplitCharges(
            r_max=CUTOFF,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            interaction_cls=mace.modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=mace.modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps("8x0e + 8x1o"),
            MLP_irreps=o3.Irreps("8x0e"),
            atomic_energies=np.zeros(len(z_table)),
            avg_num_neighbors=6.0,
            atomic_numbers=z_table.zs,
            correlation=3,
            formal_charges_from_data=False,
            atomic_formal_charges=formal_charges,
            gate=mace.modules.gate_dict["silu"],
            radial_MLP=[16, 16],
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            pbc_handling="realspace",
        )
    return model, z_table


def _batch(atoms: Atoms, reference_hessian: np.ndarray):
    dataset = dataset_from_atoms([atoms], cutoff=CUTOFF, atomic_multipoles_max_l=1)
    dataset[0].reference_hessian_flat = torch.tensor(reference_hessian.reshape(-1))
    dataset[0].molecule_assignment = torch.tensor(
        _molecule_assignment(atoms), dtype=torch.long
    )
    return torch_geometric.Batch.from_data_list(dataset)


def _forward(model, batch, compute_hessian=False):
    batch_dict = batch.to_dict()
    output = model(
        batch_dict,
        training=True,
        compute_force=True,
        compute_hessian=compute_hessian,
    )
    output["positions"] = batch_dict["positions"]
    return output


def _analytic_hessian(model, batch, atom_count) -> np.ndarray:
    output = _forward(model, batch, compute_hessian=True)
    return (
        output["hessian"]
        .detach()
        .reshape(3 * atom_count, 3 * atom_count)
        .numpy()
        .copy()
    )


def _exact_intermolecular_error(analytic, reference, assignment) -> float:
    error = analytic - reference
    coordinate_assignment = np.repeat(assignment, 3)
    total = float((error**2).sum())
    for molecule in np.unique(assignment):
        mask = coordinate_assignment == molecule
        total -= float((error[np.ix_(mask, mask)] ** 2).sum())
    return total


@pytest.mark.parametrize("target", ["full", "intermolecular"])
def test_optimising_the_loss_reduces_the_error_it_targets(target):
    """The end-to-end claim: descending this loss moves the real Hessian error down.

    The reference is the analytic Hessian of the SAME model at slightly perturbed
    weights, so the target is reachable and the minimum is near zero -- an achievable
    target is what makes "the error went down" a meaningful assertion rather than a
    statement about how expressive the model is. The error is then measured exactly,
    from the analytic Hessian, not from the estimator that is being optimised.
    """
    atoms = _water_dimer()
    model, _ = _local_split_charges_model(atoms)
    atom_count = len(atoms)
    assignment = _molecule_assignment(atoms)

    original_state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    reference = _analytic_hessian(
        model, _batch(atoms, np.zeros((3 * atom_count, 3 * atom_count))), atom_count
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(original_state[name])

    batch = _batch(atoms, reference)
    exact_before = _exact_intermolecular_error(
        _analytic_hessian(model, batch, atom_count), reference, assignment
    )

    loss_term = HessianProjectionLoss(target=target, probe_count=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    torch.manual_seed(41)
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss_term(ref=batch, pred=_forward(model, batch)).backward()
        optimizer.step()

    exact_after = _exact_intermolecular_error(
        _analytic_hessian(model, batch, atom_count), reference, assignment
    )
    assert exact_after < 0.5 * exact_before, (
        f"{target}: exact intermolecular squared error went from {exact_before:.4g} "
        f"to {exact_after:.4g}"
    )


def test_single_backward_product_matches_the_models_own_hessian():
    """One vector-Jacobian product per batch, versus the 3N-backward analytic Hessian."""
    atoms = _water_dimer()
    model, _ = _local_split_charges_model(atoms)
    atom_count = len(atoms)
    batch = _batch(atoms, np.zeros((3 * atom_count, 3 * atom_count)))

    output = _forward(model, batch, compute_hessian=True)
    analytic = output["hessian"].detach().reshape(3 * atom_count, 3 * atom_count)
    # Symmetric to machine precision: the model's second derivative is well behaved.
    assert float((analytic - analytic.T).abs().max()) < 1e-8
    # Translational invariance: the acoustic sum rule holds exactly.
    assert float(
        analytic.reshape(3 * atom_count, atom_count, 3).sum(dim=1).abs().max()
    ) < 1e-8

    probe = sample_probe(
        node_count=atom_count, device=torch.device("cpu"), dtype=torch.float64
    )
    product = hessian_vector_product(
        forces=output["forces"],
        positions=output["positions"],
        probe=probe,
        create_graph=False,
    )
    expected = (analytic @ probe.reshape(-1)).view(atom_count, 3)
    torch.testing.assert_close(product, expected, rtol=1e-9, atol=1e-10)


def test_estimator_recovers_the_analytic_hessian_error():
    """With the analytic Hessian in hand, the sketch must average to the real thing."""
    atoms = _water_dimer()
    model, _ = _local_split_charges_model(atoms)
    atom_count = len(atoms)
    generator_numpy = np.random.default_rng(3)
    reference = generator_numpy.normal(size=(3 * atom_count, 3 * atom_count))
    reference = 0.5 * (reference + reference.T)
    batch = _batch(atoms, reference)

    output = _forward(model, batch, compute_hessian=True)
    analytic = output["hessian"].detach().reshape(3 * atom_count, 3 * atom_count).numpy()
    error = analytic - reference
    exact = float((error**2).sum())

    generator = torch.Generator()
    generator.manual_seed(19)
    element_count = 9 * (batch.ptr[1:] - batch.ptr[:-1]) ** 2
    total = 0.0
    draw_count = 400
    for _ in range(draw_count):
        draw = ProjectionDraw(
            probe=sample_probe(
                node_count=atom_count,
                device=torch.device("cpu"),
                dtype=torch.float64,
                generator=generator,
            ),
            readout_mask=None,
            prefactor=1.0,
            element_count=element_count,
        )
        total += float(
            projection_estimates(
                batch=batch,
                forces=output["forces"],
                positions=output["positions"],
                draw=draw,
                create_graph=False,
            ).error_squared[0]
        )
    assert total / draw_count == pytest.approx(exact, rel=0.15)


@pytest.mark.parametrize("target", ["full", "intermolecular"])
def test_loss_gradient_flows_to_every_parameter_group(target):
    atoms = _water_dimer()
    model, _ = _local_split_charges_model(atoms)
    atom_count = len(atoms)
    batch = _batch(atoms, np.zeros((3 * atom_count, 3 * atom_count)))

    output = _forward(model, batch)
    loss = HessianProjectionLoss(target=target)(ref=batch, pred=output)
    assert loss.requires_grad
    assert float(loss) > 0.0
    loss.backward()

    with_gradient = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and float(parameter.grad.abs().max()) > 0.0
    ]
    assert with_gradient, "no parameter received a gradient from the Hessian loss"
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_loss_gradient_matches_a_finite_difference():
    """The double backward is verified against a central difference in one weight.

    The probe and the bipartition are fixed, so the loss is a deterministic function of
    the parameters and a finite difference is meaningful.
    """
    atoms = _water_dimer()
    model, _ = _local_split_charges_model(atoms)
    atom_count = len(atoms)
    generator_numpy = np.random.default_rng(4)
    reference = generator_numpy.normal(size=(3 * atom_count, 3 * atom_count)) * 0.1
    reference = 0.5 * (reference + reference.T)
    batch = _batch(atoms, reference)

    generator = torch.Generator()
    generator.manual_seed(23)
    probe = sample_probe(
        node_count=atom_count,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=generator,
    )
    receiver = (
        torch.tensor(_molecule_assignment(atoms) == 0).view(atom_count, 1)
    )
    draw = ProjectionDraw(
        probe=probe * ~receiver,
        readout_mask=receiver,
        prefactor=4.0,
        element_count=intermolecular_element_counts(batch),
    )

    def evaluate() -> torch.Tensor:
        output = _forward(model, batch)
        sample = projection_estimates(
            batch=batch,
            forces=output["forces"],
            positions=output["positions"],
            draw=draw,
            create_graph=True,
        )
        return sample.error_squared.sum() / sample.element_count.sum()

    weight = next(
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("readouts.0.linear.weight")
    )
    loss = evaluate()
    loss.backward()
    analytic_gradient = float(weight.grad.reshape(-1)[0])

    displacement = 1e-6
    with torch.no_grad():
        original = weight.reshape(-1)[0].clone()
        weight.reshape(-1)[0] = original + displacement
    forward_loss = float(evaluate())
    with torch.no_grad():
        weight.reshape(-1)[0] = original - displacement
    backward_loss = float(evaluate())
    with torch.no_grad():
        weight.reshape(-1)[0] = original

    finite_difference = (forward_loss - backward_loss) / (2 * displacement)
    assert analytic_gradient == pytest.approx(finite_difference, rel=1e-4, abs=1e-9)


def test_model_wrapper_exposes_the_differentiated_positions():
    """The loss differentiates the tensor the forward used; the wrapper hands it over."""
    atoms = _water_dimer()
    model, _ = _local_split_charges_model(atoms)
    atom_count = len(atoms)
    batch = _batch(atoms, np.zeros((3 * atom_count, 3 * atom_count)))

    wrapper = mace_scf.utils.make_model_wrapper(
        model=model,
        optimizer=None,
        output_args={
            "energy": True,
            "forces": True,
            "virials": False,
            "stress": False,
            "polarizability": False,
        },
    )
    batch_dict = batch.to_dict()
    output = wrapper(model, batch_dict, training=True)
    assert output["positions"] is batch_dict["positions"]
    loss = HessianProjectionLoss(target="full")(ref=batch, pred=output)
    assert torch.isfinite(loss)


def test_validation_metric_reports_both_targets():
    atoms = _water_dimer()
    model, _ = _local_split_charges_model(atoms)
    atom_count = len(atoms)
    generator_numpy = np.random.default_rng(8)
    reference = generator_numpy.normal(size=(3 * atom_count, 3 * atom_count)) * 0.1
    reference = 0.5 * (reference + reference.T)

    dataset = dataset_from_atoms([atoms], cutoff=CUTOFF, atomic_multipoles_max_l=1)
    dataset[0].reference_hessian_flat = torch.tensor(reference.reshape(-1))
    dataset[0].molecule_assignment = torch.tensor(
        _molecule_assignment(atoms), dtype=torch.long
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False
    )
    wrapper = mace_scf.utils.make_model_wrapper(
        model=model,
        optimizer=None,
        output_args={
            "energy": True,
            "forces": True,
            "virials": False,
            "stress": False,
            "polarizability": False,
        },
    )

    from mace_scf.hessian_projections import evaluate_hessian_projections

    metrics = evaluate_hessian_projections(
        model=model,
        model_eval_wrapper=wrapper,
        data_loader=loader,
        device=torch.device("cpu"),
        probe_count=4,
        seed=1,
    )
    assert set(metrics) == {
        "rmse_hessian_full",
        "rel_hessian_full",
        "rmse_hessian_intermolecular",
        "rel_hessian_intermolecular",
    }
    assert all(np.isfinite(value) and value > 0.0 for value in metrics.values())
