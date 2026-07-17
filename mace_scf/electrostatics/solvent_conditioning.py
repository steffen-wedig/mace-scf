"""Solvent (dielectric) conditioning of node features for implicit solvation.

design_doc.md A3: inject the dielectric constant into the node embedding as a *bounded*
encoding ``1 - 1/eps in [0, 1)`` (not raw eps, whose scale of 78 would dominate the feature
space), so that the solvent context propagates into the interactions, the energy readouts,
and the ``lr_source_maps`` that predict the multipoles. Energy-only conditioning is
insufficient: both the strained-density intramolecular energy and the predicted multipoles
shift between phases.

The dielectric constant is a per-graph scalar carried on the batch as
``data["dielectric_constant"]`` (see ``ExtAtomicData``); it is broadcast to nodes via the
batch index before being passed here.

The polynomial basis ``[feature, feature^2, ...]`` vanishes at ``feature = 0`` (gas,
eps = 1) and the embedding linear carries no bias, so a solvent-conditioned model reduces
*exactly* to its gas behaviour when eps = 1 -- there is no spurious constant shift in
vacuum. This keeps the "gas reproduces gas DFT with the field off" constraint (A4) intact.

These modules mirror the ``oxidation_state_mixer`` pattern in ``bonded_blocks.py`` and are
TorchScript-scriptable (the model classes use ``@compile_mode("script")``).
"""

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode


@compile_mode("script")
class SolventFeatureEmbeddingBlock(torch.nn.Module):
    """Embed the bounded dielectric feature ``1 - 1/eps`` into invariant node features.

    The output has the scalar (l = 0) irreps of ``node_feats_irreps`` so it can be added
    into the node features. It is identically zero when eps = 1 (gas).
    """

    exponents: torch.Tensor

    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        num_basis: int = 4,
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        invariant_node_feats = o3.Irreps(
            f"{node_feats_irreps.count(o3.Irrep(0, 1))}x0e"
        )
        # biases=False so the embedding is zero when the feature (and thus the basis) is
        # zero, i.e. in the gas phase.
        self.solvent_feature_linear = o3.Linear(
            o3.Irreps(f"{num_basis}x0e"), invariant_node_feats, biases=False
        )
        self.register_buffer(
            "exponents",
            torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()),
        )

    @staticmethod
    def bounded_dielectric_feature(dielectric_constant: torch.Tensor) -> torch.Tensor:
        """Map eps -> 1 - 1/eps in [0, 1). Gas (eps = 1) -> 0; conductor (eps -> inf) -> 1."""
        return 1.0 - 1.0 / dielectric_constant

    def forward(self, node_dielectric_constant: torch.Tensor) -> torch.Tensor:
        # node_dielectric_constant: [n_nodes] eps per node (broadcast from per-graph eps).
        feature = self.bounded_dielectric_feature(node_dielectric_constant).unsqueeze(-1)
        # polynomial basis [feature, feature^2, ...], each term vanishing at feature = 0.
        basis = feature**self.exponents  # [n_nodes, num_basis]
        return self.solvent_feature_linear(basis)


class NoSolventConditioning(torch.nn.Module):
    """Identity mixer: no dielectric conditioning (gas-only models, Exp 0)."""

    def __init__(self, node_feats_irreps: o3.Irreps, num_basis: int = 4) -> None:
        super().__init__()

    def forward(
        self,
        node_feats: torch.Tensor,
        node_dielectric_constant: torch.Tensor,
    ) -> torch.Tensor:
        return node_feats


class SumSolventConditioning(torch.nn.Module):
    """Add the embedded dielectric feature into the (scalar part of the) node features."""

    def __init__(self, node_feats_irreps: o3.Irreps, num_basis: int = 4) -> None:
        super().__init__()
        self.solvent_embedding = SolventFeatureEmbeddingBlock(
            node_feats_irreps=node_feats_irreps, num_basis=num_basis
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_dielectric_constant: torch.Tensor,
    ) -> torch.Tensor:
        return node_feats + self.solvent_embedding(node_dielectric_constant)


# Registry mirroring `oxidation_state_mixers` in bonded_blocks.py; selected from config.
solvent_conditioning_mixers = {
    "none": NoSolventConditioning,
    "sum": SumSolventConditioning,
}


def build_solvent_conditioning(
    name: str,
    node_feats_irreps: o3.Irreps,
    num_basis: int = 4,
) -> torch.nn.Module:
    """Construct a solvent-conditioning mixer by name (config-driven)."""
    if name not in solvent_conditioning_mixers:
        raise ValueError(
            f"Unknown solvent_conditioning {name!r}; "
            f"choices: {sorted(solvent_conditioning_mixers)}"
        )
    return solvent_conditioning_mixers[name](
        node_feats_irreps=node_feats_irreps, num_basis=num_basis
    )


__all__ = [
    "SolventFeatureEmbeddingBlock",
    "NoSolventConditioning",
    "SumSolventConditioning",
    "solvent_conditioning_mixers",
    "build_solvent_conditioning",
]
