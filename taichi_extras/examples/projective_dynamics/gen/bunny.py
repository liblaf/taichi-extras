from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pyvista as pv
import typer
from pymeshfix import MeshFix
from pyvista import PolyData, examples

from taichi_extras.io import smesh
from taichi_extras.pyvista import poly_data

CENTER: tuple[float, float, float] = (0.01, -0.03, 0.01)


def main(
    output: Annotated[
        Path,
        typer.Argument(
            exists=False, file_okay=True, dir_okay=False, writable=True, readable=False
        ),
    ]
) -> None:
    inner_mesh: PolyData = cast(
        PolyData, pv.Icosphere(radius=0.03, center=CENTER, nsub=3)
    )
    outer_mesh: PolyData = cast(PolyData, examples.download_bunny())
    mesh_fix: MeshFix = MeshFix(outer_mesh)
    mesh_fix.repair(verbose=True, remove_smallest_components=True)
    outer_mesh = mesh_fix.mesh
    outer_mesh.points -= outer_mesh.center

    inner_points, inner_indices = poly_data.get_vertices_indices(inner_mesh)
    outer_points, outer_indices = poly_data.get_vertices_indices(outer_mesh)
    points: np.ndarray = np.concatenate([inner_points, outer_points])
    indices: np.ndarray = np.concatenate(
        [inner_indices, outer_indices + len(inner_points)]
    )
    holes: np.ndarray = np.array([list(CENTER)])
    smesh.write(output, points=points, facets=indices, holes=holes)


if __name__ == "__main__":
    typer.run(main)
