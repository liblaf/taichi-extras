import typing
from pathlib import Path

import numpy as np
import pyvista as pv
import typer

import taichi_extras.tetgen.io.smesh


def main(
    output_filepath: Path = typer.Argument(
        ..., exists=False, file_okay=True, dir_okay=False, writable=True, readable=False
    ),
) -> None:
    mesh: pv.PolyData = typing.cast(pv.PolyData, pv.Cube())
    mesh: pv.PolyData = typing.cast(pv.PolyData, mesh.triangulate())
    points: np.ndarray = mesh.points
    indices: np.ndarray = mesh.faces.copy().reshape(-1, 4)[:, 1:4]
    taichi_extras.tetgen.io.smesh.write(output_filepath, points=points, facets=indices)


if __name__ == "__main__":
    typer.run(main)
