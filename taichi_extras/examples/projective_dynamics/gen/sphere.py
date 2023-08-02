import typing
from pathlib import Path
from typing import Annotated

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
    mesh: PolyData = typing.cast(PolyData, pv.Sphere())
    points, indices = poly_data.get_vertices_indices(mesh)
    smesh.write(output, points=points, facets=indices)


if __name__ == "__main__":
    typer.run(main)
