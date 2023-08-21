from pathlib import Path
from typing import Annotated

import taichi as ti
import typer
from taichi import MatrixField, MeshInstance

from taichi_extras.io import node
from taichi_extras.io import pyvista as io_pv
from taichi_extras.utils.mesh import element_field

ti.init()


def main(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(
            exists=False, file_okay=True, dir_okay=False, writable=True, readable=False
        ),
    ],
) -> None:
    mesh: MeshInstance = node.read_mesh(input, relations=["FV"])
    element_field.place_safe(field=mesh.verts, members={"position": ti.math.vec3})
    position: MatrixField = mesh.verts.get_member_field("position")
    position.from_numpy(mesh.get_position_as_numpy())
    io_pv.write_mesh(output, mesh)


if __name__ == "__main__":
    typer.run(main)
