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
    ],
    *,
    nsub: Annotated[int, typer.Option("-s", "--sub")] = 3,
) -> None:
    inner_sphere: PolyData = typing.cast(PolyData, pv.Icosphere(radius=0.5, nsub=nsub))
    outer_sphere: PolyData = typing.cast(PolyData, pv.Icosphere(radius=1.0, nsub=nsub))
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
