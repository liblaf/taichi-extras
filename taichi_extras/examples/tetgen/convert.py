from pathlib import Path
from typing import Annotated

import taichi as ti
import typer

from taichi_extras.io import node, pyvista

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
    *,
    binary: Annotated[bool, typer.Option("--binary/--ascii")] = True,
) -> None:
    mesh: ti.MeshInstance = node.read_mesh(input, relations=["FV"])
    pyvista.write_mesh(output, mesh=mesh, binary=binary)


if __name__ == "__main__":
    typer.run(main)
