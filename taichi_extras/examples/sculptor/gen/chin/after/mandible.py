from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pyvista import PolyData

from taichi_extras.io import pyvista as io_pv

THRESHOLD: np.ndarray = np.array([np.nan, -55.0, np.nan])


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
    indices: np.ndarray = mesh.points[:, 1] < THRESHOLD[1]
    mesh.points[indices, 2] += 10.0  # type: ignore
    mesh.save(output)


if __name__ == "__main__":
    typer.run(main)
