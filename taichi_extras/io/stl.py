from pathlib import Path

import numpy as np
import taichi as ti
from taichi import MatrixField, MeshInstance
from trimesh import Trimesh

from taichi_extras.utils.mesh import element_field


def write(filepath: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Parameters:
        vertices: (n, 3)
        faces   : (m, 3) or (m, 4)
    """
    tri_mesh: Trimesh = Trimesh(vertices=vertices, faces=faces)
    tri_mesh.export(filepath)


@ti.kernel
def get_faces_kernel(mesh: ti.template()):  # type: ignore
    for f in mesh.faces:
        f.indices = [f.verts[i].id for i in range(3)]


def get_faces(mesh: ti.MeshInstance) -> np.ndarray:
    element_field.place_safe(
        field=mesh.faces, members={"indices": ti.types.vector(n=3, dtype=ti.i32)}
    )
    get_faces_kernel(mesh=mesh)
    indices: ti.MatrixField = mesh.faces.get_member_field("indices")
    return indices.to_numpy()


def write_mesh(filepath: Path, mesh: MeshInstance, key: str = "position") -> None:
    position: np.ndarray
    if key and key in mesh.verts.keys:
        position_field: MatrixField = mesh.verts.get_member_field("position")
        position = position_field.to_numpy()
    else:
        position = mesh.get_position_as_numpy()
    indices: np.ndarray = get_faces(mesh=mesh)
    write(filepath=filepath, vertices=position, faces=indices)
