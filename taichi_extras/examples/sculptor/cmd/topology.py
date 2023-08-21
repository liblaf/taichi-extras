from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pyvista as pv
import typer
from pyvista import PolyData


def get_num_vertices(filepath: Path) -> int:
    mesh: PolyData = cast(PolyData, pv.read(filepath))
    return mesh.n_points


def main(
    meshes: Annotated[
        list[Path],
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    *,
    output: Annotated[
        Path,
        typer.Option(
            exists=False, file_okay=True, dir_okay=False, writable=True, readable=False
        ),
    ],
) -> None:
    num_vertices: np.ndarray = np.array(list(map(get_num_vertices, meshes)))
    cum_sum_vertices: np.ndarray = np.cumsum(num_vertices)
    np.savetxt(output, cum_sum_vertices, fmt="%d")


if __name__ == "__main__":
    typer.run(main)
