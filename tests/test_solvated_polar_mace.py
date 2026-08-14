"""Integration tests for SolvatedPolarMACE (PolarMACE + in-loop continuum reaction field).

Checks the wiring end-to-end on the real model:
* gas (eps = 1) reproduces the base PolarMACE energy exactly (reaction field is zero -> no
  double counting, no perturbation; the subclass adds no learnable parameters);
* a solvated (eps > 1) forward shifts the energy and, for the polarizable generalized-Kirkwood
  tier, changes the predicted density (the solvent re-polarises the solute -- the whole point);
* forces run under training; the field_feature_max_l >= 1 guard fires.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from e3nn import o3
from mace.data.utils import Configuration
from mace.modules import PolarMACE, interaction_classes
from mace.tools import AtomicNumberTable, torch_geometric
from mace_scf.data.new_atomic_data import ExtAtomicData
from mace_scf.electrostatics.solvated_polar_mace import SolvatedPolarMACE

torch.set_default_dtype(torch.float64)

Z_TABLE = AtomicNumberTable([1, 8])


def _backbone_config() -> dict:
    return dict(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(2),
        avg_num_neighbors=2.0,
        atomic_numbers=[1, 8],
        correlation=2,
        gate=torch.nn.functional.silu,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        kspace_cutoff_factor=1.5,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.5,
        field_feature_max_l=1,
        field_feature_widths=[1.5],
        num_recursion_steps=2,
    )


def _build_base_polar_mace() -> torch.nn.Module:
    torch.manual_seed(0)
    return PolarMACE(**_backbone_config())


def _build_solvated(scheme: str = "generalized_kirkwood") -> torch.nn.Module:
    torch.manual_seed(0)
    return SolvatedPolarMACE(**_backbone_config(), reaction_field_scheme=scheme)


def _water_batch(dielectric_constant: float):
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


def test_field_feature_max_l_guard():
    config = _backbone_config()
    config["field_feature_max_l"] = 0
    with pytest.raises(ValueError, match="field_feature_max_l"):
        SolvatedPolarMACE(**config, reaction_field_scheme="generalized_kirkwood")


def test_subclass_adds_no_learnable_parameters():
    # provider + projection carry only buffers, so gas parity below is a clean comparison.
    base = _build_base_polar_mace()
    solvated = _build_solvated("generalized_kirkwood")
    base_names = {n for n, _ in base.named_parameters()}
    solvated_names = {n for n, _ in solvated.named_parameters()}
    assert base_names == solvated_names


@pytest.mark.parametrize("scheme", ["generalized_born", "generalized_kirkwood", "ddcosmo"])
def test_gas_equals_base_polar_mace(scheme):
    base = _build_base_polar_mace()
    solvated = _build_solvated(scheme)
    batch = _water_batch(1.0)
    base_energy = base(batch, training=False)["energy"]
    solvated_energy = solvated(_water_batch(1.0), training=False)["energy"]
    assert torch.allclose(base_energy, solvated_energy, atol=1e-10)


@pytest.mark.parametrize("scheme", ["generalized_born", "generalized_kirkwood", "ddcosmo"])
def test_solvent_shifts_energy(scheme):
    solvated = _build_solvated(scheme)
    gas_energy = solvated(_water_batch(1.0), training=False)["energy"]
    water_energy = solvated(_water_batch(78.3553), training=False)["energy"]
    assert not torch.allclose(gas_energy, water_energy, atol=1e-6)


def test_generalized_kirkwood_repolarises_density():
    # The mechanism: with in-loop feedback the predicted density (hence dipole) responds to the
    # solvent. Gas vs water density coefficients must differ for the Kirkwood tier.
    solvated = _build_solvated("generalized_kirkwood")
    gas = solvated(_water_batch(1.0), training=False)["density_coefficients"]
    water = solvated(_water_batch(78.3553), training=False)["density_coefficients"]
    assert not torch.allclose(gas, water, atol=1e-6)


@pytest.mark.parametrize("scheme", ["generalized_born", "generalized_kirkwood", "ddcosmo"])
def test_forces_run_under_training(scheme):
    solvated = _build_solvated(scheme)
    output = solvated(_water_batch(78.3553), training=True, compute_force=True)
    forces = output["forces"]
    assert forces.shape == (3, 3)
    assert torch.isfinite(forces).all()
    assert torch.isfinite(output["energy"]).all()
