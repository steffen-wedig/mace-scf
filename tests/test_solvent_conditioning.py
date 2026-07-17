"""Tests for the dielectric (solvent) conditioning modules (design_doc.md A3).

The conditioning is injected right after the node embedding, where the node features are
scalar-only (``node_feats_irreps = "{n}x0e"`` in LocalSplitCharges), so these tests use a
scalar node-feature irreps, matching the real injection point.
"""

import torch
from e3nn import o3
from mace_scf.electrostatics.solvent_conditioning import (
    NoSolventConditioning,
    SolventFeatureEmbeddingBlock,
    SumSolventConditioning,
    build_solvent_conditioning,
)

NODE_FEATS_IRREPS = o3.Irreps("16x0e")


def _node_features(num_nodes: int) -> torch.Tensor:
    torch.manual_seed(0)
    return NODE_FEATS_IRREPS.randn(num_nodes, -1)


def test_bounded_feature_limits():
    eps = torch.tensor([1.0, 2.0, 78.3553, 1.0e6])
    feature = SolventFeatureEmbeddingBlock.bounded_dielectric_feature(eps)
    assert feature[0].item() == 0.0  # gas
    assert 0.0 < feature[1].item() < 1.0
    # feature = 1 - 1/eps is in [0, 1) mathematically; the large-eps end approaches 1.
    assert torch.all(feature >= 0.0) and torch.all(feature <= 1.0)
    assert feature[2].item() < 1.0


def test_sum_conditioning_is_identity_in_gas():
    # eps = 1 -> bounded feature 0 -> polynomial basis 0 -> zero shift (no bias).
    mixer = SumSolventConditioning(NODE_FEATS_IRREPS)
    for parameter in mixer.parameters():
        torch.nn.init.normal_(parameter, std=0.5)  # nonzero weights; gas must still be identity
    node_feats = _node_features(5)
    gas_eps = torch.ones(5)
    out = mixer(node_feats, gas_eps)
    torch.testing.assert_close(out, node_feats)


def test_sum_conditioning_shifts_in_solvent():
    mixer = SumSolventConditioning(NODE_FEATS_IRREPS)
    for parameter in mixer.parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    node_feats = _node_features(5)
    water_eps = torch.full((5,), 78.3553)
    out = mixer(node_feats, water_eps)
    assert not torch.allclose(out, node_feats)
    # a higher dielectric produces a different (generally larger-magnitude) shift than a low one
    low_eps = torch.full((5,), 2.02)
    shift_water = (mixer(node_feats, water_eps) - node_feats).abs().mean()
    shift_low = (mixer(node_feats, low_eps) - node_feats).abs().mean()
    assert shift_water > shift_low


def test_no_conditioning_is_identity():
    mixer = NoSolventConditioning(NODE_FEATS_IRREPS)
    node_feats = _node_features(4)
    out = mixer(node_feats, torch.full((4,), 78.3553))
    torch.testing.assert_close(out, node_feats)


def test_build_by_name():
    assert isinstance(build_solvent_conditioning("none", NODE_FEATS_IRREPS), NoSolventConditioning)
    assert isinstance(build_solvent_conditioning("sum", NODE_FEATS_IRREPS), SumSolventConditioning)


def test_torchscript_scriptable():
    mixers = (NoSolventConditioning(NODE_FEATS_IRREPS), SumSolventConditioning(NODE_FEATS_IRREPS))
    for mixer in mixers:
        scripted = torch.jit.script(mixer)
        node_feats = _node_features(3)
        eps = torch.tensor([1.0, 20.5, 78.3553])
        torch.testing.assert_close(scripted(node_feats, eps), mixer(node_feats, eps))
