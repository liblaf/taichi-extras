from pathlib import Path
from typing import Annotated

import numpy as np
import taichi as ti
import typer
from trimesh import Trimesh

from taichi_extras.io import node
from taichi_extras.utils.mesh import element_field

ti.init()


@ti.kernel
def get_indices_kernel(mesh: ti.template()):  # type: ignore
    for f in mesh.faces:
        f.indices = [f.verts[i].id for i in range(3)]


def get_indices(mesh: ti.MeshInstance) -> np.ndarray:
    element_field.place_safe(
        field=mesh.faces, members={"indices": ti.types.vector(n=3, dtype=ti.i32)}
    )
    get_indices_kernel(mesh=mesh)
    indices: ti.MatrixField = mesh.faces.get_member_field("indices")
    return indices.to_numpy()


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
    mesh: ti.MeshInstance = node.read(input, relations=["FV"])
    indices: np.ndarray = get_indices(mesh)
    tri_mesh: Trimesh = Trimesh(vertices=mesh.get_position_as_numpy(), faces=indices)
    tri_mesh.export(output)


if __name__ == "__main__":
    typer.run(main)
