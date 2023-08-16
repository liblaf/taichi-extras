from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pyvista as pv
import taichi as ti
import typer
from matplotlib.cm import ScalarMappable
from pyvista import PolyData


@ti.kernel
def diff(
    old: ti.template(),  # type: ignore
    new: ti.template(),  # type: ignore
) -> None:
    for v in new.verts:
        v.displacement = ti.math.length(v.position - old.verts[v.id].position)


def main(
    old: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    new: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    *,
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
        ),
    ]
) -> None:
    old_mesh: PolyData = cast(PolyData, pv.read(old))
    new_mesh: PolyData = cast(PolyData, pv.read(new))
    assert old_mesh.n_points == new_mesh.n_points
    displacement: np.ndarray = np.linalg.norm(new_mesh.points - old_mesh.points, axis=1)
    displacement /= max(np.max(displacement), 1e-6)  # type: ignore
    scalar_mappable: ScalarMappable = ScalarMappable()
    rgba: np.ndarray = cast(
        np.ndarray, scalar_mappable.to_rgba(displacement, bytes=True)
    )
    rgb: np.ndarray = np.delete(rgba, -1, axis=-1)
    new_mesh.save(output, texture=rgb)


if __name__ == "__main__":
    typer.run(main)
