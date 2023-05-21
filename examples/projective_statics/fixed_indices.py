from pathlib import Path
from typing import Optional

import numpy as np
import taichi as ti
import typer

import taichi_extras.tetgen.io.results

ti.init()


def main(
    mesh_filepath: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
    ),
    output_filepath: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=False,
    ),
    radius: float = typer.Option(0.5, "-r", "--radius", min=0.0),
) -> None:
    if output_filepath is None:
        output_filepath = mesh_filepath.with_suffix(".fixed.txt")
    mesh, faces = taichi_extras.tetgen.io.results.read(filepath=mesh_filepath)
    position: np.ndarray = mesh.get_position_as_numpy()
    fixed: np.ndarray = np.sum(position**2, axis=1) < ((radius**2) + 1e-6)
    np.savetxt(fname=output_filepath, X=fixed, fmt="%d")


if __name__ == "__main__":
    typer.run(main)
