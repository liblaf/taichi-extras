from pathlib import Path
from typing import Annotated

import typer
from pyvista import PolyData

from taichi_extras.io import pyvista as io_pv


def main(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, readable=True, writable=False
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(
            exists=False, file_okay=True, dir_okay=False, readable=False, writable=True
        ),
    ],
) -> None:
    mesh: PolyData = io_pv.read_poly_data(input)
    mesh.points[:, 1] = mesh.center[1] + (mesh.points[:, 1] - mesh.center[1]) * 2.0  # type: ignore
    mesh.save(output)


if __name__ == "__main__":
    typer.run(main)
