import typing
from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
import typer

from taichi_extras.io import smesh


def main(
    output: Annotated[
        Path,
        typer.Argument(
            exists=False, file_okay=True, dir_okay=False, writable=True, readable=False
        ),
    ]
) -> None:
    inner_sphere: pv.PolyData = typing.cast(pv.PolyData, pv.Sphere(radius=0.5))
    outer_sphere: pv.PolyData = typing.cast(pv.PolyData, pv.Sphere(radius=1.0))
    inner_points: np.ndarray = inner_sphere.points
    outer_points: np.ndarray = outer_sphere.points
    inner_indices: np.ndarray = inner_sphere.faces.copy().reshape(-1, 4)[:, 1:4]
    outer_indices: np.ndarray = outer_sphere.faces.copy().reshape(-1, 4)[:, 1:4]

    points: np.ndarray = np.concatenate([inner_points, outer_points])
    for i in range(inner_indices.shape[0]):
        continue
    for i in range(outer_indices.shape[0]):
        outer_indices[i] += inner_points.shape[0]
    indices: np.ndarray = np.concatenate([inner_indices, outer_indices])

    holes: np.ndarray = np.array([[0.0, 0.0, 0.0]])

    smesh.write(output, points=points, facets=indices, holes=holes)


if __name__ == "__main__":
    typer.run(main)
