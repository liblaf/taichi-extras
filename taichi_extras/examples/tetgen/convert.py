from pathlib import Path
from typing import Annotated

import taichi as ti
import typer

from taichi_extras.io import node, stl

ti.init()


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
    mesh: ti.MeshInstance = node.read_mesh(input, relations=["FV"])
    stl.write_mesh(output, mesh=mesh)


if __name__ == "__main__":
    typer.run(main)
