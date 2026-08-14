"""Optimizer parameter-group coverage for get_param_options.

Every trainable parameter of a model must be assigned to an optimizer
parameter group, otherwise it silently stays at its random initialization
for the whole training run (this happened to all PolarMACE field modules,
and to LocalSplitCharges' oxidation_state_mixer).
"""

import argparse

import numpy as np
import pytest
import torch
from e3nn import o3

import mace.modules

import mace_scf.electrostatics
from mace_scf.utils.run_train_utils import get_param_options


def minimal_args(model_name: str) -> argparse.Namespace:
    """The subset of parsed training arguments that get_param_options reads."""
    return argparse.Namespace(
        model=model_name,
        lr=0.01,
        weight_decay=5e-7,
        amsgrad=True,
        beta=0.9,
        beta_two=0.999,
        local_charges_weight_decay=5e-7,
        field_block_weight_decay=5e-7,
    )


def backbone_config() -> dict:
    return dict(
        r_max=3.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=mace.modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=mace.modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(2),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=2,
        gate=torch.nn.functional.silu,
    )


def build_tiny_local_split_charges() -> torch.nn.Module:
    return mace_scf.electrostatics.LocalSplitCharges(
        **backbone_config(),
        formal_charges_from_data=True,
        atomic_multipoles_max_l=0,
        atomic_multipoles_smearing_width=1.5,
        static_bond_transfer_block="OxidationDependentSymmetricPredictionSourceBlock",
        oxidation_state_mixer="SumOxidationStateMixer",
        oxidation_state_range=(-6.0, 6.0),
    )


def build_tiny_polar_mace() -> torch.nn.Module:
    return mace.modules.PolarMACE(
        **backbone_config(),
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        kspace_cutoff_factor=1.5,
        atomic_multipoles_max_l=0,
        atomic_multipoles_smearing_width=1.5,
        field_feature_max_l=0,
        field_feature_widths=[1.5],
        num_recursion_steps=1,
    )


def build_tiny_solvated_polar_mace() -> torch.nn.Module:
    return mace_scf.electrostatics.SolvatedPolarMACE(
        **backbone_config(),
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        kspace_cutoff_factor=1.5,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.5,
        field_feature_max_l=1,
        field_feature_widths=[1.5],
        num_recursion_steps=2,
        reaction_field_scheme="generalized_kirkwood",
    )


def assert_full_parameter_coverage(model: torch.nn.Module, args: argparse.Namespace):
    param_options = get_param_options(model, args)

    parameters_in_groups = {
        id(parameter)
        for group in param_options["params"]
        for parameter in group["params"]
    }
    uncovered = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in parameters_in_groups
    ]
    assert uncovered == []

    # The groups must also form a valid (duplicate-free) optimizer input.
    torch.optim.AdamW(**param_options)
    return param_options


def test_local_split_charges_param_options_cover_all_parameters():
    model = build_tiny_local_split_charges()
    param_options = assert_full_parameter_coverage(model, minimal_args("LocalSplitCharges"))
    group_names = {group["name"] for group in param_options["params"]}
    assert "oxidation_state_mixer" in group_names


def test_polar_mace_param_options_cover_all_parameters():
    if not hasattr(mace.modules, "PolarMACE"):
        pytest.skip("PolarMACE requires mace-torch >= 0.3.16")
    model = build_tiny_polar_mace()
    param_options = assert_full_parameter_coverage(model, minimal_args("PolarMACE"))
    group_names = {group["name"] for group in param_options["params"]}
    assert {
        "lr_source_maps",
        "fukui_source_map",
        "field_dependent_charges_maps",
        "layer_feature_mixer",
    } <= group_names


def test_solvated_polar_mace_param_options_cover_all_parameters():
    if not hasattr(mace.modules, "PolarMACE"):
        pytest.skip("SolvatedPolarMACE requires mace-torch >= 0.3.16 (PolarMACE base)")
    model = build_tiny_solvated_polar_mace()
    param_options = assert_full_parameter_coverage(
        model, minimal_args("SolvatedPolarMACE")
    )
    group_names = {group["name"] for group in param_options["params"]}
    assert {
        "lr_source_maps",
        "fukui_source_map",
        "field_dependent_charges_maps",
        "layer_feature_mixer",
    } <= group_names


def test_unassigned_parameters_raise():
    model = build_tiny_local_split_charges()
    model.rogue_module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="rogue_module"):
        get_param_options(model, minimal_args("LocalSplitCharges"))
