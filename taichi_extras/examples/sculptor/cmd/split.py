from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import typer
from matplotlib.cm import ScalarMappable
from pyvista import PolyData

from taichi_extras.io import node as io_node
from taichi_extras.io import pyvista as io_pv


def run(old: PolyData, new: np.ndarray, output: Path) -> None:
    displacement: np.ndarray = np.linalg.norm(new - old.points, axis=1)
    mappable: ScalarMappable = ScalarMappable()
    rgba: np.ndarray = cast(np.ndarray, mappable.to_rgba(displacement, bytes=True))
    old.points = new
    old.save(output, texture=rgba)


def main(
    outer: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    inner: Annotated[
        list[Path],
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    node: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    topology: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, dir_okay=True, writable=True, readable=False
        ),
    ],
) -> None:
    topo: np.ndarray = np.loadtxt(topology, dtype=int)
    outer_mesh: PolyData = io_pv.read_poly_data(outer)
    inner_meshes: Sequence[PolyData] = list(map(io_pv.read_poly_data, inner))
    assert topo.shape == (1 + len(inner_meshes),)
    position: np.ndarray = io_node.read(node)
    run(old=outer_mesh, new=position[: topo[0]], output=output / "outer.ply")
    for i, mesh in enumerate(inner_meshes):
        begin: int = topo[i]
        end: int = topo[i + 1]
        run(old=mesh, new=position[begin:end], output=output / "inner" / f"{i:02d}.ply")


if __name__ == "__main__":
    typer.run(main)
