from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
import typer
from pyvista import PolyData

from taichi_extras.io import pyvista as io_pv
from taichi_extras.io import smesh
from taichi_extras.pyvista import poly_data


def main(
    outer: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    inner: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    *,
    output: Annotated[Path, typer.Option(dir_okay=False, writable=True, readable=False)]
) -> None:
    outer_mesh: PolyData = io_pv.read_poly_data(outer)
    inner_meshes: list[PolyData] = list(map(io_pv.read_poly_data, inner))
    holes: np.ndarray = np.array(list(map(poly_data.find_enclosed_point, inner_meshes)))
    all_mesh: PolyData = pv.merge([outer_mesh, *inner_meshes], main_has_priority=False)
    points, faces = poly_data.get_vertices_indices(all_mesh)
    smesh.write(output, points=points, facets=faces, holes=holes)


if __name__ == "__main__":
    typer.run(main)
