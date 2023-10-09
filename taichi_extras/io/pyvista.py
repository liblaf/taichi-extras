from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
import taichi as ti
from pyvista import PolyData
from taichi import MatrixField, MeshInstance

from taichi_extras.lang.mesh import mesh_element_field


def read_poly_data(filepath: Path) -> PolyData:
    mesh = pv.read(filepath)
    assert isinstance(mesh, PolyData)
    return mesh


def write(
    filepath: Path,
    points: np.ndarray,
    faces: np.ndarray,
    *,
    binary: bool = True,
    texture: Optional[str | np.ndarray] = None
) -> None:
    mesh: PolyData = pv.make_tri_mesh(points=points, faces=faces)
    mesh.save(filepath, binary=binary, texture=texture)


@ti.kernel
def get_faces_kernel(mesh: ti.template()):  # type: ignore
    for f in mesh.faces:
        f.faces = [f.verts[i].id for i in range(3)]


def get_faces(mesh: MeshInstance) -> np.ndarray:
    mesh_element_field.place(
        field=mesh.faces, members={"faces": ti.types.vector(n=3, dtype=ti.i32)}
    )
    get_faces_kernel(mesh=mesh)
    faces: MatrixField = mesh.faces.get_member_field("faces")
    return faces.to_numpy()


def write_mesh(
    filepath: Path,
    mesh: MeshInstance,
    key: str = "position",
    *,
    binary: bool = True,
    texture: Optional[str | np.ndarray] = None
) -> None:
    points: np.ndarray
    if key and (key in mesh.verts.keys):
        position_field: MatrixField = mesh.verts.get_member_field("position")
        points = position_field.to_numpy()
    else:
        points = mesh.get_position_as_numpy()
    faces: np.ndarray = get_faces(mesh=mesh)
    write(filepath=filepath, points=points, faces=faces, binary=binary, texture=texture)
