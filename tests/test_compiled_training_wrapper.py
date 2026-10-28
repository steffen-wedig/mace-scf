"""Compiled training forwards of LocalSplitCharges match eager ones.

CompiledTrainingWrapper runs the training forward through torch.compile in one
graph with dynamic shapes. A random-weight model on open water clusters of
changing sizes, with the analytical real-space electrostatics, must give the same
loss and parameter gradients as the eager wrapper, also after an eager
evaluation forward has switched the parameters' gradients off. The aot_eager
backend traces exactly what inductor would compile but needs no C++ toolchain.
"""

import numpy as np
import pytest
import torch
from ase import build
from e3nn import o3

import mace.data
import mace.modules
import mace.tools
from mace.tools import torch_geometric, torch_tools

import mace_scf.data
from mace_scf.electrostatics import LocalSplitCharges
from mace_scf.utils.model_training_wrappers import (
    CompiledTrainingWrapper,
    LocalSourcesModelWrapper,
    configure_training_compilation,
)
from tests.utils import disable_e3nn_codegen, seed_torch

R_MAX = 3.0
OUTPUT_ARGUMENTS = {"forces": True, "virials": False, "stress": False}


def build_model(z_table) -> LocalSplitCharges:
    torch_tools.set_default_dtype("float64")
    seed_torch(3)
    interaction_cls = mace.modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    with disable_e3nn_codegen():
        model = LocalSplitCharges(
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
            atomic_energies=np.array([-12.1, -2057.9]),
            avg_num_neighbors=5.0,
            atomic_numbers=z_table.zs,
            correlation=2,
            formal_charges_from_data=False,
            atomic_formal_charges=np.array([1.0, -2.0]),
            gate=mace.modules.gate_dict["silu"],
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            include_electrostatic_self_interaction=True,
            pbc_handling="realspace",
            realspace_method="analytical",
        )
    return model


def water_cluster(molecules: int, seed: int):
    generator = np.random.default_rng(seed)
    water = build.molecule("H2O")
    cluster = water.copy()
    for _ in range(molecules - 1):
        molecule = water.copy()
        molecule.rotate(generator.uniform(0.0, 360.0), "z")
        molecule.translate(generator.uniform(-3.0, 3.0, size=3))
        cluster += molecule
    return cluster


def batch_dict(z_table, sizes, seed):
    dataset = [
        mace_scf.data.ExtAtomicData.from_config(
            mace.data.config_from_atoms(water_cluster(size, seed + index)),
            z_table=z_table,
            cutoff=R_MAX,
            atomic_multipoles_max_l=1,
        )
        for index, size in enumerate(sizes)
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False
    )
    return next(iter(loader)).to_dict()


def loss_and_gradients(wrapper, model, batch):
    output = wrapper(model, batch, training=True)
    loss = output["energy"].pow(2).sum() + output["forces"].pow(2).sum()
    gradients = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    return loss, gradients


@pytest.mark.parametrize("sizes", [[2, 3], [1, 4, 2], [3, 3, 3]])
def test_compiled_training_forward_matches_eager(sizes):
    configure_training_compilation()
    z_table = mace.tools.get_atomic_number_table_from_zs([1, 8])
    model = build_model(z_table)
    eager = LocalSourcesModelWrapper(optimizer=None, output_args=OUTPUT_ARGUMENTS)
    compiled = CompiledTrainingWrapper(eager, backend="aot_eager")

    batch = batch_dict(z_table, sizes, seed=sum(sizes))
    # An eager evaluation first: it switches the parameters' gradients off,
    # which the compiled training call must undo.
    compiled(model, batch_dict(z_table, sizes, seed=1), training=False)
    compiled_loss, compiled_gradients = loss_and_gradients(compiled, model, batch)
    eager_loss, eager_gradients = loss_and_gradients(
        eager, model, batch_dict(z_table, sizes, seed=sum(sizes))
    )

    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-12, atol=1e-12)
    # The traced graphs live in the compiled view only; the model is what is
    # saved, with its own module types.
    assert not isinstance(model.products[0].symmetric_contractions, torch.fx.GraphModule)
    assert not isinstance(model.readouts[-1], torch.fx.GraphModule)
    for compiled_gradient, eager_gradient in zip(compiled_gradients, eager_gradients):
        if eager_gradient is None:
            assert compiled_gradient is None
            continue
        torch.testing.assert_close(
            compiled_gradient, eager_gradient, rtol=1e-10, atol=1e-12
        )


def test_compiled_training_rejects_unsupported_wrappers():
    class OtherWrapper:
        optimizer = None
        output_args = OUTPUT_ARGUMENTS

    with pytest.raises(ValueError, match="compiled training supports"):
        CompiledTrainingWrapper(OtherWrapper())
