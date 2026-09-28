"""Tests for the TorchSim interface of LocalSplitCharges (upstream: tests/test_torchsim.py).

A random-weight model built on the fly (as in test_lsc_stress_finite_difference.py) is
compared with the ASE calculator MACELocalSplitCharges system by system, on CPU with the
pure-torch neighbour list, so no GPU, checkpoint or nvalchemiops is needed.
"""

import numpy as np
import pytest
import torch
from ase.atoms import Atoms

import mace.modules  # before e3nn: mace sets the torch.load policy e3nn's import needs
import mace.tools
from e3nn import get_optimization_defaults, o3, set_optimization_defaults
from mace.tools import torch_tools

ts = pytest.importorskip("torch_sim")
from torch_sim.neighbors import torch_nl_linked_cell  # noqa: E402

from mace_scf.calculators.localsources import MACELocalSplitCharges  # noqa: E402
from mace_scf.data.neighborhood import get_neighborhood  # noqa: E402
from mace_scf.electrostatics import LocalSplitCharges  # noqa: E402

try:
    from mace_scf.calculators.torchsim import (
        LocalSplitChargesTorchSimModel,
        nonperiodic_cell,
    )
except ImportError:  # prototype location (water_hessians experiments/torchsim_lsc)
    from torchsim import LocalSplitChargesTorchSimModel, nonperiodic_cell

FORMAL_CHARGES_KEY = "formal_oxidation_states"
ENERGY_TOLERANCE = 1e-9  # eV; observed ~1e-13
FORCE_TOLERANCE = 1e-9  # eV/A; observed ~1e-14
STRESS_TOLERANCE = 1e-11  # eV/A^3; observed ~1e-17

TRICLINIC_CELL = np.array([[7.0, 0.0, 0.0], [1.3, 6.8, 0.0], [0.9, -1.1, 6.9]])
WATER_DIMER = np.array(
    [
        [3.10, 3.05, 3.12],
        [3.65, 3.80, 2.95],
        [2.31, 3.36, 3.55],
        [4.85, 5.10, 4.60],
        [5.25, 4.35, 5.05],
        [4.25, 5.55, 5.22],
    ]
)


def build_random_model(pbc_handling: str) -> LocalSplitCharges:
    torch_tools.set_default_dtype("float64")
    torch.manual_seed(42)
    z_table = mace.tools.get_atomic_number_table_from_zs([1, 8])
    interaction_cls = mace.modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    codegen = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    try:
        return LocalSplitCharges(
            r_max=3.0,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls,
            num_interactions=2,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps("8x0e+8x1o"),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.array([1.0] * len(z_table)),
            avg_num_neighbors=10.0,
            atomic_numbers=z_table.zs,
            correlation=2,
            formal_charges_from_data=True,
            gate=mace.modules.gate_dict["silu"],
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            include_electrostatic_self_interaction=True,
            pbc_handling=pbc_handling,
        )
    finally:
        set_optimization_defaults(jit_script_fx=codegen)


def water(positions, cell, pbc, charged=False):
    n = len(positions) // 3
    atoms = Atoms(symbols="OHH" * n, positions=positions, cell=cell, pbc=pbc)
    charges = np.tile([-2.0, 1.0, 1.0], n)
    if charged:
        charges[1] = 2.0  # net +1: exercises the formal-charge input
    atoms.arrays[FORMAL_CHARGES_KEY] = charges
    return atoms


def periodic_systems():
    rng = np.random.default_rng(0)
    shifted = WATER_DIMER + np.array([[1, 0, 0], [1, 0, 0], [0, -1, 0], [-1, 1, 0], [0, 0, 1], [3, 0, -1]]) @ TRICLINIC_CELL
    tetramer = np.concatenate([WATER_DIMER, WATER_DIMER + [2.9, 0.4, -1.8]])
    return [
        water(WATER_DIMER, TRICLINIC_CELL, True),
        water(shifted + rng.normal(0, 0.05, shifted.shape), TRICLINIC_CELL * 1.05, True),
        water(tetramer, np.diag([8.0, 7.5, 9.0]), True),
    ]


def cluster_systems():
    trimer = np.concatenate([WATER_DIMER, WATER_DIMER[:3] + [0.2, 2.8, 0.5]])
    systems = [
        water(WATER_DIMER, np.zeros((3, 3)), False),
        water(trimer, np.zeros((3, 3)), False, charged=True),
    ]
    # atoms_to_state reads an info key only if EVERY system carries it.
    systems[0].info["external_field"] = np.zeros(3)
    systems[1].info["external_field"] = np.array([0.01, -0.02, 0.005])
    return systems


def to_state(atoms_list, model):
    return ts.io.atoms_to_state(
        atoms_list,
        model.device,
        model.dtype,
        atom_extras_map={"formal_charges": FORMAL_CHARGES_KEY},
        system_extras_map={"external_E_field": "external_field"},
    )


@pytest.fixture(scope="module", params=["pbc", "realspace"])
def models(request, tmp_path_factory):
    model = build_random_model(request.param)
    path = tmp_path_factory.mktemp("models") / f"lsc_{request.param}.model"
    torch.save(model, path)
    calculator = MACELocalSplitCharges(
        model_path=str(path),
        device="cpu",
        formal_charges_key=FORMAL_CHARGES_KEY,
        pbc_handling=request.param,
    )
    wrapper = LocalSplitChargesTorchSimModel(
        str(path), device="cpu", neighbor_list_fn=torch_nl_linked_cell
    )
    systems = periodic_systems() if request.param == "pbc" else cluster_systems()
    return calculator, wrapper, systems


def test_batch_matches_ase_calculator(models):
    calculator, wrapper, systems = models
    torch.set_default_dtype(torch.float32)  # the wrapper must not rely on the global
    try:
        out = wrapper(to_state(systems, wrapper))
    finally:
        torch.set_default_dtype(torch.float64)
    offset = 0
    for index, atoms in enumerate(systems):
        atoms = atoms.copy()
        atoms.calc = calculator
        n = len(atoms)
        assert abs(out["energy"][index].item() - atoms.get_potential_energy()) < ENERGY_TOLERANCE
        np.testing.assert_allclose(
            out["forces"][offset : offset + n].numpy(), atoms.get_forces(), rtol=0, atol=FORCE_TOLERANCE
        )
        np.testing.assert_allclose(
            out["stress"][index].numpy(), atoms.get_stress(voigt=False), rtol=0, atol=STRESS_TOLERANCE
        )
        offset += n


def test_system_alone_matches_system_in_batch(models):
    _, wrapper, systems = models
    batch = wrapper(to_state(systems, wrapper))
    n0 = len(systems[0])
    alone = wrapper(to_state(systems[1:2], wrapper))
    torch.testing.assert_close(alone["energy"][0], batch["energy"][1], rtol=0, atol=1e-10)
    torch.testing.assert_close(
        alone["forces"], batch["forces"][n0 : n0 + len(systems[1])], rtol=0, atol=1e-10
    )


def test_outputs_are_detached_with_expected_shapes(models):
    _, wrapper, systems = models
    state = to_state(systems, wrapper)
    out = wrapper(state)
    assert out["energy"].shape == (state.n_systems,)
    assert out["forces"].shape == (state.n_atoms, 3)
    assert out["stress"].shape == (state.n_systems, 3, 3)
    assert not any(value.requires_grad for value in out.values())
    assert not state.positions.requires_grad


def test_short_nvt_is_finite(models):
    _, wrapper, systems = models
    state = to_state(systems, wrapper)
    state.rng = 0
    final = ts.integrate(
        system=state,
        model=wrapper,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=5,
        temperature=300.0,
        timestep=0.0005,
    )
    assert torch.isfinite(final.positions).all()


@pytest.mark.parametrize("scale", [0.3, 12.0])
def test_nonperiodic_cell_matches_get_neighborhood(scale):
    positions = [WATER_DIMER * scale, WATER_DIMER[:3]]
    cells = [np.zeros((3, 3)), np.diag([40.0, 10.0, 35.0])]
    expected = [
        get_neighborhood(p, cutoff=3.0, pbc=(False, False, False), cell=c.copy())[3]
        for p, c in zip(positions, cells)
    ]
    system_idx = torch.tensor([0] * len(positions[0]) + [1] * len(positions[1]))
    got = nonperiodic_cell(
        torch.tensor(np.stack(cells)),
        torch.tensor(np.concatenate(positions)),
        system_idx,
    )
    np.testing.assert_allclose(got.numpy(), np.stack(expected), rtol=0, atol=1e-12)
