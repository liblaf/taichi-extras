import typing
from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
import typer
from pyvista import PolyData

from taichi_extras.io import smesh
from taichi_extras.pyvista import poly_data


def main(
    output: Annotated[
        Path,
        typer.Argument(
            exists=False, file_okay=True, dir_okay=False, writable=True, readable=False
        ),
    ]
) -> None:
    inner_sphere: PolyData = typing.cast(
        PolyData, pv.Sphere(radius=0.5, direction=(0.0, 1.0, 0.0))
    )
    outer_sphere: PolyData = typing.cast(
        PolyData, pv.Sphere(radius=1.0, direction=(0.0, 1.0, 0.0))
    )
    inner_points, inner_indices = poly_data.get_vertices_indices(inner_sphere)
    outer_points, outer_indices = poly_data.get_vertices_indices(outer_sphere)
    points: np.ndarray = np.concatenate([inner_points, outer_points])
    indices: np.ndarray = np.concatenate(
        [inner_indices, outer_indices + len(inner_points)]
    )
    holes: np.ndarray = np.array([[0.0, 0.0, 0.0]])
    smesh.write(output, points=points, facets=indices, holes=holes)


if __name__ == "__main__":
    typer.run(main)
