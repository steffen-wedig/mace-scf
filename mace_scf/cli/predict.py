"""Energies, forces and total charges of a trained LocalSplitCharges model.

Installed as the ``mace-scf-predict`` console script:

    mace-scf-predict --model <stage>.model --structures frames.xyz \
        --output predictions.npz --device cuda --default-dtype float64

The structures go through exactly the data path the trainer uses
(``mace.data.config_from_atoms`` -> ``mace_scf.data.ExtAtomicData`` -> batched
``torch_geometric`` loader -> ``model(batch, compute_force=True)``), in batches,
so a prediction reproduces the energy the trainer's error table saw. The
output archive holds, in the input order:

* ``energies`` -- total energy per structure [eV], shape (structures,)
* ``forces`` -- forces concatenated over structures [eV/Angstrom], (atoms, 3)
* ``total_charges`` -- the summed monopoles of the predicted charge density
  [e], shape (structures,); with fixed formal charges, the composition's charge
* ``atom_counts`` -- atoms per structure, shape (structures,)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PredictionRequest:
    """What to predict, with which model, where."""

    model_path: Path
    structures_path: Path
    output_path: Path
    device: Literal["cpu", "cuda"] = "cuda"
    default_dtype: Literal["float32", "float64"] = "float64"
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be cpu or cuda, not {self.device!r}")
        if self.default_dtype not in ("float32", "float64"):
            raise ValueError(
                f"default_dtype must be float32 or float64, not {self.default_dtype!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, not {self.batch_size}")


def predict(request: PredictionRequest) -> None:
    """Run the model over every structure and write the archive.

    Raises:
        FileNotFoundError: if the model or the structures file is missing.
        ValueError: if the structures file is empty or holds an element the
            model was not trained on.
    """
    if not request.model_path.is_file():
        raise FileNotFoundError(f"model not found: {request.model_path}")
    if not request.structures_path.is_file():
        raise FileNotFoundError(f"structures not found: {request.structures_path}")

    import ase.io
    import mace.data
    import mace_scf.data
    import numpy as np
    import torch
    from mace.tools import torch_geometric, torch_tools, utils

    torch_tools.set_default_dtype(request.default_dtype)
    device = torch_tools.init_device(request.device)
    model = torch.load(request.model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    frames = list(ase.io.iread(request.structures_path, index=":"))
    if not frames:
        raise ValueError(f"no structures in {request.structures_path}")
    model_numbers = {int(number) for number in model.atomic_numbers}
    for index, frame in enumerate(frames):
        unknown = sorted({int(number) for number in frame.numbers} - model_numbers)
        if unknown:
            raise ValueError(
                f"structure {index} holds elements {unknown} the model was not "
                f"trained on ({sorted(model_numbers)})"
            )

    z_table = utils.AtomicNumberTable([int(number) for number in model.atomic_numbers])
    cutoff = float(model.r_max)
    multipoles_max_l = int(model.coulomb_energy.density_max_l)
    heads = list(getattr(model, "heads", ["Default"]))
    # No labels are read: the charge comes from the formal charges, and the
    # keys of the training labels must not leak into the inputs.
    key_specification = mace.data.KeySpecification(info_keys={}, arrays_keys={})
    dataset = [
        mace_scf.data.ExtAtomicData.from_config(
            mace.data.config_from_atoms(
                frame, key_specification=key_specification, head_name=heads[0]
            ),
            z_table=z_table,
            cutoff=cutoff,
            atomic_multipoles_max_l=multipoles_max_l,
            heads=heads,
        )
        for frame in frames
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=request.batch_size, shuffle=False, drop_last=False
    )

    energies: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    total_charges: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), training=False, compute_force=True)
        energies.append(output["energy"].detach().cpu().numpy())
        forces.append(output["forces"].detach().cpu().numpy())
        monopoles = output["density_coefficients"][:, 0]
        total_charges.append(
            torch.zeros(batch.num_graphs, dtype=monopoles.dtype, device=monopoles.device)
            .index_add_(0, batch["batch"], monopoles)
            .detach()
            .cpu()
            .numpy()
        )

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        request.output_path,
        energies=np.concatenate(energies),
        forces=np.concatenate(forces),
        total_charges=np.concatenate(total_charges),
        atom_counts=np.array([len(frame) for frame in frames]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args(argv)
    predict(
        PredictionRequest(
            model_path=arguments.model,
            structures_path=arguments.structures,
            output_path=arguments.output,
            device=arguments.device,
            default_dtype=arguments.default_dtype,
            batch_size=arguments.batch_size,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
