from pathlib import Path
from typing import Annotated

import numpy as np
import taichi as ti
import typer
from taichi import MatrixField, MeshInstance, Vector

from taichi_extras.io import node
from taichi_extras.utils.mesh import element_field

ti.init()


@ti.kernel
def fix_kernel(mesh: ti.template()):  # type: ignore
    for v in mesh.verts:
        if (ti.abs(v.position) > 0.49).all():  # type: ignore
            v.fixed = v.position
            v.fixed.y *= 1.5
        else:
            v.fixed = Vector([np.nan, np.nan, np.nan])


def fix(mesh: MeshInstance):
    element_field.place_safe(
        field=mesh.verts, members={"fixed": ti.math.vec3, "position": ti.math.vec3}
    )
    position: MatrixField = mesh.verts.get_member_field("position")
    position.from_numpy(mesh.get_position_as_numpy())
    fix_kernel(mesh=mesh)


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
    mesh: MeshInstance = node.read_mesh(input, relations=["CV"])
    fix(mesh=mesh)
    fixed: MatrixField = mesh.verts.get_member_field("fixed")
    node.write(output, fixed.to_numpy())


if __name__ == "__main__":
    typer.run(main)
