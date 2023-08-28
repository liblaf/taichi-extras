from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pyvista import PolyData

from taichi_extras.io import pyvista as io_pv

THRESHOLD: np.ndarray = np.array([np.nan, -40.0, 65.0])


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
    indices: np.ndarray = (mesh.points[:, 1] < THRESHOLD[1]) & (
        mesh.points[:, 2] > THRESHOLD[2]
    )
    mesh.points[indices, 2] += 0.5 * (  # type: ignore
        np.min(
            np.array(
                [
                    THRESHOLD[1] - mesh.points[indices, 1],
                    mesh.points[indices, 2] - THRESHOLD[2],
                ]
            ),
            axis=0,
        )
    )
    mesh.save(output)


if __name__ == "__main__":
    typer.run(main)
