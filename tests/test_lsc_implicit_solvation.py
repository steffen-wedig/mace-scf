"""Integration tests for implicit solvation wired into LocalSplitCharges.

Checks the design_doc.md A4 decomposition end-to-end on the real model:
* gas (eps = 1) with solvation enabled == gas with solvation disabled (reaction field is
  exactly zero, so gas reproduces gas DFT with the field off -- no double counting);
* a solvated (eps > 1) forward shifts the total energy by a nonzero reaction-field term;
* one weight set reproduces gas (field off) and solution (field on).
"""

import numpy as np
import torch
from e3nn import o3
from mace.data.utils import Configuration
from mace.modules import interaction_classes
from mace.tools import AtomicNumberTable, torch_geometric
from mace_scf.data.new_atomic_data import ExtAtomicData
from mace_scf.electrostatics.localsources import LocalSplitCharges

torch.set_default_dtype(torch.float64)

Z_TABLE = AtomicNumberTable([1, 8])


def _build_model(enable_implicit_solvation, reaction_field_scheme="screened_gb",
                 solvent_conditioning="none"):
    torch.manual_seed(0)
    return LocalSplitCharges(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(2),
        avg_num_neighbors=2.0,
        atomic_numbers=[1, 8],
        correlation=2,
        formal_charges_from_data=False,
        gate=torch.nn.functional.silu,
        atomic_formal_charges=np.zeros(2),
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.5,
        pbc_handling="realspace",
        enable_implicit_solvation=enable_implicit_solvation,
        reaction_field_scheme=reaction_field_scheme,
        solvent_conditioning=solvent_conditioning,
    )


def _water_batch(dielectric_constant):
    config = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]),
        properties={"dielectric_constant": dielectric_constant, "total_charge": 0.0},
        property_weights={},
        pbc=(False, False, False),
        cell=np.zeros((3, 3)),
    )
    data = ExtAtomicData.from_config(config, Z_TABLE, cutoff=5.0, atomic_multipoles_max_l=1)
    loader = torch_geometric.dataloader.DataLoader([data], batch_size=1)
    return next(iter(loader)).to_dict()


def test_gas_forward_matches_disabled_solvation():
    model_on = _build_model(enable_implicit_solvation=True)
    model_off = _build_model(enable_implicit_solvation=False)
    # identical weights
    model_off.load_state_dict(model_on.state_dict(), strict=False)

    energy_on = model_on(_water_batch(1.0), training=False)["energy"]
    energy_off = model_off(_water_batch(1.0), training=False)["energy"]
    torch.testing.assert_close(energy_on, energy_off)


def test_solvated_forward_shifts_energy():
    model = _build_model(enable_implicit_solvation=True)
    gas_energy = model(_water_batch(1.0), training=False)["energy"]
    water_energy = model(_water_batch(78.3553), training=False)["energy"]
    # the reaction field is a nonzero (stabilizing) shift in solvent
    assert not torch.allclose(gas_energy, water_energy)


def test_gas_energy_independent_of_scheme():
    # eps = 1 -> both schemes add exactly zero, so the gas energy is scheme-independent
    gb = _build_model(enable_implicit_solvation=True, reaction_field_scheme="screened_gb")
    ddcosmo = _build_model(enable_implicit_solvation=True, reaction_field_scheme="ddcosmo")
    ddcosmo.load_state_dict(gb.state_dict(), strict=False)
    gb_gas = gb(_water_batch(1.0), training=False)["energy"]
    ddcosmo_gas = ddcosmo(_water_batch(1.0), training=False)["energy"]
    torch.testing.assert_close(gb_gas, ddcosmo_gas)


def test_forces_flow_through_reaction_field():
    model = _build_model(enable_implicit_solvation=True)
    output = model(_water_batch(78.3553), training=True, compute_force=True)
    forces = output["forces"]
    assert forces is not None
    assert forces.shape == (3, 3)
    assert torch.isfinite(forces).all()


def test_conditioning_changes_solvated_prediction():
    # With "sum" conditioning and nonzero embedding weights, the predicted multipoles must
    # differ between gas and solvent (the network sees the dielectric context), while the
    # conditioning still vanishes at eps = 1 by construction.
    conditioned = _build_model(enable_implicit_solvation=True, solvent_conditioning="sum")
    for parameter in conditioned.solvent_conditioning.parameters():
        torch.nn.init.normal_(parameter, std=0.3)

    gas_multipoles = conditioned(_water_batch(1.0), training=False)["density_coefficients"]
    water_multipoles = conditioned(
        _water_batch(78.3553), training=False
    )["density_coefficients"]
    assert torch.isfinite(gas_multipoles).all()
    assert not torch.allclose(gas_multipoles, water_multipoles)
