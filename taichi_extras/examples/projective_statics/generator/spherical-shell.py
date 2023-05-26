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
    inner_sphere: pv.PolyData = typing.cast(
        pv.PolyData,
        pv.Sphere(direction=(0.0, 1.0, 0.0)),
    )
    outer_sphere: pv.PolyData = typing.cast(
        pv.PolyData, pv.Sphere(radius=1.0, direction=(0.0, 1.0, 0.0))
    )
    inner_points: np.ndarray = inner_sphere.points
    outer_points: np.ndarray = outer_sphere.points
    inner_indices: np.ndarray = inner_sphere.faces.copy().reshape(-1, 4)[:, 1:4]
    outer_indices: np.ndarray = outer_sphere.faces.copy().reshape(-1, 4)[:, 1:4]

    points: np.ndarray = np.concatenate([inner_points, outer_points])
    if False:
        for i in range(inner_indices.shape[0]):
            inner_indices[i] = inner_indices[i, ::-1]
    for i in range(outer_indices.shape[0]):
        outer_indices[i] += inner_points.shape[0]
    indices: np.ndarray = np.concatenate([inner_indices, outer_indices])

    holes: np.ndarray = np.array([[0.0, 0.0, 0.0]])

    taichi_extras.tetgen.io.smesh.write(
        output_filepath, points=points, facets=indices, holes=holes
    )


if __name__ == "__main__":
    typer.run(main)
