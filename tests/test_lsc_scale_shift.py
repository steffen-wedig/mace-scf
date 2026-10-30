"""Per-head scale and shift of the LocalSplitCharges interaction energy.

The model mirrors mace.modules.ScaleShiftMACE: every interaction layer's node
energies are multiplied by the head's scale and the head's shift is added once
per atom, to the first interaction layer, so that

    E = E0 + E_Coulomb + scale[head] * E_inter + shift[head] * N_atoms

and the per-layer contributions still sum to the total minus the Coulomb
energy. The defaults (scale 1, shift 0) are an exact no-op.
"""

import copy  # noqa: I001 (mace must be imported before e3nn, see below)

import numpy as np
import pytest
import torch
from ase import build

# mace before e3nn: importing mace sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which
# e3nn's constants load needs on torch >= 2.6.
import mace.data
import mace.modules
import mace.tools
from mace.tools import torch_geometric, torch_tools
from mace.tools.scatter import scatter_sum
from e3nn import o3

import mace_scf.data
from mace_scf.calculators.localsources import MACELocalSplitCharges
from mace_scf.electrostatics import LocalSplitCharges
from mace_scf.electrostatics.compiled_localsources import (
    build_compiled_local_source_evaluator,
)
from mace_scf.utils.extend_arg_parse import extended_arg_parser
from mace_scf.utils.model_training_wrappers import (
    CompiledTrainingWrapper,
    LocalSourcesModelWrapper,
    configure_training_compilation,
)
from mace_scf.utils.run_train_utils import build_model, parse_scale_shift_argument
from tests.utils import disable_e3nn_codegen, seed_torch

R_MAX = 3.0
HEADS = ["first", "second"]
SCALES = [2.5, 0.4]
SHIFTS = [-0.3, 1.7]


def build_lsc_model(z_table, pbc_handling="realspace", **kwargs) -> LocalSplitCharges:
    torch_tools.set_default_dtype("float64")
    seed_torch(5)
    interaction_cls = mace.modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    with disable_e3nn_codegen():
        return LocalSplitCharges(
            r_max=R_MAX,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls,
            num_interactions=2,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps("8x0e+8x1o"),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.array([[-12.1, -2057.9], [-13.4, -2050.2]]),
            avg_num_neighbors=5.0,
            atomic_numbers=z_table.zs,
            correlation=2,
            formal_charges_from_data=False,
            atomic_formal_charges=np.array([1.0, -2.0]),
            gate=mace.modules.gate_dict["silu"],
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            include_electrostatic_self_interaction=True,
            pbc_handling=pbc_handling,
            realspace_method="analytical" if pbc_handling == "realspace" else "finite_difference",
            heads=HEADS,
            **kwargs,
        )


def with_scale_shift(model: LocalSplitCharges, scale, shift) -> LocalSplitCharges:
    """A copy of ``model`` with the same weights and new per-head scale/shift."""
    scaled = copy.deepcopy(model)
    scaled.scale_shift.scale = torch.tensor(scale, dtype=torch.float64)
    scaled.scale_shift.shift = torch.tensor(shift, dtype=torch.float64)
    return scaled


def water_cluster(molecules: int, seed: int, periodic: bool = False):
    generator = np.random.default_rng(seed)
    water = build.molecule("H2O")
    cluster = water.copy()
    for _ in range(molecules - 1):
        molecule = water.copy()
        molecule.rotate(generator.uniform(0.0, 360.0), "z")
        molecule.translate(generator.uniform(-3.0, 3.0, size=3))
        cluster += molecule
    if periodic:
        cluster.set_cell([9.0, 9.5, 10.0])
        cluster.center()
        cluster.pbc = True
    return cluster


def batch_dict(z_table, sizes, head_names, periodic=False):
    dataset = [
        mace_scf.data.ExtAtomicData.from_config(
            mace.data.config_from_atoms(
                water_cluster(size, 11 + index, periodic=periodic), head_name=head_name
            ),
            z_table=z_table,
            cutoff=R_MAX,
            atomic_multipoles_max_l=1,
            heads=HEADS,
        )
        for index, (size, head_name) in enumerate(zip(sizes, head_names, strict=True))
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False
    )
    return next(iter(loader)).to_dict()


def evaluate(model, z_table, sizes, head_names):
    batch = batch_dict(z_table, sizes, head_names)
    output = model(batch, compute_force=True)
    return batch, output


def raw_readout_energies(model, batch):
    """Per-layer, per-graph readout energies before scale/shift, via hooks."""
    captured = []
    hooks = [
        readout.register_forward_hook(lambda module, inputs, output: captured.append(output))
        for readout in model.readouts
    ]
    try:
        model(batch, compute_force=False)
    finally:
        for hook in hooks:
            hook.remove()
    node_heads = batch["head"][batch["batch"]]
    num_graphs = batch["ptr"].numel() - 1
    return [
        scatter_sum(
            src=torch.gather(output, dim=1, index=node_heads.view(-1, 1)).squeeze(-1),
            index=batch["batch"],
            dim=-1,
            dim_size=num_graphs,
        )
        for output in captured
    ]


SIZES = [2, 3, 4]
HEAD_NAMES = ["first", "second", "first"]


@pytest.fixture(name="z_table")
def fixture_z_table():
    return mace.tools.get_atomic_number_table_from_zs([1, 8])


def test_default_scale_shift_is_an_exact_no_op(z_table):
    model = build_lsc_model(z_table)
    # A float default is broadcast to every head.
    assert torch.equal(model.scale_shift.scale, torch.ones(len(HEADS), dtype=torch.float64))
    assert torch.equal(model.scale_shift.shift, torch.zeros(len(HEADS), dtype=torch.float64))
    _, output = evaluate(model, z_table, SIZES, HEAD_NAMES)
    raw = raw_readout_energies(model, batch_dict(z_table, SIZES, HEAD_NAMES))
    # Every interaction contribution is the bare readout sum, bit for bit.
    for layer_index, raw_energy in enumerate(raw):
        assert torch.equal(output["contributions"][:, layer_index + 1], raw_energy)

    # Explicit per-head ones and zeros give identical energies and forces.
    explicit = build_lsc_model(
        z_table, atomic_inter_scale=[1.0, 1.0], atomic_inter_shift=[0.0, 0.0]
    )
    _, explicit_output = evaluate(explicit, z_table, SIZES, HEAD_NAMES)
    assert torch.equal(explicit_output["energy"], output["energy"])
    # Forces only to round-off: the CPU backward reductions are not bitwise
    # reproducible between two evaluations.
    torch.testing.assert_close(
        explicit_output["forces"], output["forces"], rtol=0.0, atol=1e-12
    )


def test_scale_shift_gives_scaled_interaction_energy_per_head(z_table):
    model = build_lsc_model(z_table)
    without_interaction = with_scale_shift(model, [0.0, 0.0], [0.0, 0.0])
    scaled = with_scale_shift(model, SCALES, SHIFTS)

    _, unit_output = evaluate(model, z_table, SIZES, HEAD_NAMES)
    _, base_output = evaluate(without_interaction, z_table, SIZES, HEAD_NAMES)
    batch, scaled_output = evaluate(scaled, z_table, SIZES, HEAD_NAMES)

    head_index = batch["head"]
    scale = torch.tensor(SCALES, dtype=torch.float64)[head_index]
    shift = torch.tensor(SHIFTS, dtype=torch.float64)[head_index]
    num_atoms = (batch["ptr"][1:] - batch["ptr"][:-1]).to(torch.float64)

    # E0 + Coulomb: the energy with the interaction switched off.
    e0_plus_coulomb = base_output["energy"]
    interaction_energy = unit_output["energy"] - e0_plus_coulomb
    torch.testing.assert_close(
        scaled_output["energy"],
        e0_plus_coulomb + scale * interaction_energy + shift * num_atoms,
        rtol=1e-12,
        atol=1e-10,
    )
    # E0 is not scaled.
    torch.testing.assert_close(
        scaled_output["contributions"][:, 0], unit_output["contributions"][:, 0]
    )
    # Per-layer contributions: every layer scaled, the shift added once (layer 0).
    node_scale = scale[batch["batch"]]
    raw = raw_readout_energies(model, batch_dict(z_table, SIZES, HEAD_NAMES))
    for layer_index, raw_energy in enumerate(raw):
        expected = scale * raw_energy + (shift * num_atoms if layer_index == 0 else 0.0)
        torch.testing.assert_close(
            scaled_output["contributions"][:, layer_index + 1],
            expected,
            rtol=1e-12,
            atol=1e-10,
        )
    # Node energies keep summing to the local part of the total.
    torch.testing.assert_close(
        scatter_sum(scaled_output["node_energy"], batch["batch"], dim=-1),
        scaled_output["contributions"].sum(dim=-1),
    )
    # Forces: the Coulomb forces are unchanged, the interaction forces scale.
    base_forces = base_output["forces"]
    torch.testing.assert_close(
        scaled_output["forces"],
        base_forces + node_scale.unsqueeze(-1) * (unit_output["forces"] - base_forces),
        rtol=1e-10,
        atol=1e-10,
    )


def test_scale_shift_round_trips_through_torch_save(z_table, tmp_path):
    scaled = with_scale_shift(build_lsc_model(z_table), SCALES, SHIFTS)
    path = tmp_path / "scaled.model"
    torch.save(scaled, path)
    loaded = torch.load(path, weights_only=False)
    assert torch.equal(loaded.scale_shift.scale, scaled.scale_shift.scale)
    assert torch.equal(loaded.scale_shift.shift, scaled.scale_shift.shift)
    _, expected = evaluate(scaled, z_table, SIZES, HEAD_NAMES)
    _, actual = evaluate(loaded, z_table, SIZES, HEAD_NAMES)
    assert torch.equal(actual["energy"], expected["energy"])
    # The buffers are part of the state dict.
    state = scaled.state_dict()
    assert torch.equal(state["scale_shift.scale"], scaled.scale_shift.scale)
    assert torch.equal(state["scale_shift.shift"], scaled.scale_shift.shift)


def test_calculator_loads_the_scale_shift(z_table, tmp_path):
    model = build_lsc_model(
        z_table, pbc_handling="pbc", atomic_inter_scale=SCALES, atomic_inter_shift=SHIFTS
    )
    path = tmp_path / "scaled.model"
    torch.save(model, path)
    calculator = MACELocalSplitCharges(
        model_path=str(path), device="cpu", pbc_handling="pbc"
    )
    assert torch.equal(calculator.model.scale_shift.scale, model.scale_shift.scale)
    assert torch.equal(calculator.model.scale_shift.shift, model.scale_shift.shift)

    atoms = water_cluster(3, 11, periodic=True)
    atoms.calc = calculator
    # The calculator evaluates the first head.
    reference = model(batch_dict(z_table, [3], ["first"], periodic=True), compute_force=True)
    np.testing.assert_allclose(
        atoms.get_potential_energy(), reference["energy"].item(), rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        atoms.get_forces(), reference["forces"].numpy(), rtol=1e-10, atol=1e-10
    )


def test_scale_shift_rejects_wrong_number_of_heads(z_table):
    with pytest.raises(ValueError, match="atomic_inter_scale has 3 entries"):
        build_lsc_model(z_table, atomic_inter_scale=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="atomic_inter_shift has 1 entries"):
        build_lsc_model(z_table, atomic_inter_shift=[1.0])


@pytest.mark.parametrize("compiled", [False, True])
def test_compiled_core_matches_eager_with_scale_shift(z_table, compiled):
    model = build_lsc_model(
        z_table,
        pbc_handling="pbc",
        atomic_inter_scale=SCALES,
        atomic_inter_shift=SHIFTS,
    )
    batch = batch_dict(z_table, SIZES, HEAD_NAMES, periodic=True)
    reference = model(batch, compute_force=True)
    evaluator = build_compiled_local_source_evaluator(
        model,
        pbc_handling="pbc",
        backend="aot_eager",
        mode="default",
        enabled=compiled,
    )
    actual = evaluator.evaluate(batch_dict(z_table, SIZES, HEAD_NAMES, periodic=True))
    for key in ("energy", "node_energy", "forces", "density_coefficients", "dipole"):
        torch.testing.assert_close(actual[key], reference[key], rtol=1e-10, atol=1e-10)
    # The scale/shift is actually active in this comparison.
    unscaled = with_scale_shift(model, [1.0, 1.0], [0.0, 0.0])
    unscaled_reference = unscaled(
        batch_dict(z_table, SIZES, HEAD_NAMES, periodic=True), compute_force=True
    )
    assert not torch.allclose(unscaled_reference["energy"], reference["energy"])


@pytest.mark.parametrize(
    "value, expected",
    [("1.0", 1.0), ("2", 2.0), ("[2.5, 0.4]", [2.5, 0.4]), ("(1, -3.5)", [1.0, -3.5])],
)
def test_parse_scale_shift_argument(value, expected):
    assert parse_scale_shift_argument(value, "--atomic_inter_scale") == expected


@pytest.mark.parametrize("value", ["abc", "[]", "True", "{'a': 1.0}", "['x']"])
def test_parse_scale_shift_argument_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="--atomic_inter_shift"):
        parse_scale_shift_argument(value, "--atomic_inter_shift")


def lsc_arguments(*extra):
    return extended_arg_parser().parse_args(
        [
            "--name", "scale_shift_test",
            "--train_schedule", "[]",
            "--heads", "{}",
            "--error_table", "PerAtomRMSE",
            "--model", "LocalSplitCharges",
            "--num_interactions", "2",
            "--hidden_irreps", "8x0e+8x1o",
            "--MLP_irreps", "16x0e",
            "--max_ell", "2",
            "--r_max", str(R_MAX),
            "--avg_num_neighbors", "5.0",
            "--atomic_formal_charges", "{1: 1.0, 8: -2.0}",
            "--atomic_multipoles_max_l", "1",
            *extra,
        ]
    )


def test_cli_defaults_are_a_no_op(z_table):
    args = lsc_arguments()
    assert args.atomic_inter_scale == "1.0"
    assert args.atomic_inter_shift == "0.0"


def test_cli_scale_shift_reaches_the_model(z_table):
    args = lsc_arguments("--atomic_inter_scale", "2.5", "--atomic_inter_shift", "-0.3")
    if not hasattr(args, "compute_polarizability"):
        args.compute_polarizability = False
    torch_tools.set_default_dtype("float64")
    with disable_e3nn_codegen():
        model = build_model(
            args=args,
            z_table=z_table,
            atomic_energies=np.array([-12.1, -2057.9]),
            atomic_charges=np.array([1.0, -2.0]),
            train_loader=None,
        )
    assert isinstance(model, LocalSplitCharges)
    assert model.heads == ["Default"]
    assert torch.equal(model.scale_shift.scale, torch.tensor([2.5], dtype=torch.float64))
    assert torch.equal(model.scale_shift.shift, torch.tensor([-0.3], dtype=torch.float64))


def test_compiled_training_forward_matches_eager_with_scale_shift(z_table):
    configure_training_compilation()
    model = build_lsc_model(
        z_table, atomic_inter_scale=SCALES, atomic_inter_shift=SHIFTS
    )
    output_arguments = {"forces": True, "virials": False, "stress": False}
    eager = LocalSourcesModelWrapper(optimizer=None, output_args=output_arguments)
    compiled = CompiledTrainingWrapper(eager, backend="aot_eager")

    def loss_and_gradients(wrapper):
        output = wrapper(model, batch_dict(z_table, SIZES, HEAD_NAMES), training=True)
        loss = output["energy"].pow(2).sum() + output["forces"].pow(2).sum()
        gradients = torch.autograd.grad(
            loss, list(model.parameters()), allow_unused=True
        )
        return loss, gradients

    compiled_loss, compiled_gradients = loss_and_gradients(compiled)
    eager_loss, eager_gradients = loss_and_gradients(eager)
    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-12, atol=1e-12)
    for compiled_gradient, eager_gradient in zip(
        compiled_gradients, eager_gradients, strict=True
    ):
        if eager_gradient is None:
            assert compiled_gradient is None
            continue
        torch.testing.assert_close(
            compiled_gradient, eager_gradient, rtol=1e-10, atol=1e-12
        )
