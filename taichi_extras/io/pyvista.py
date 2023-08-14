from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
import taichi as ti
from pyvista import PolyData
from taichi import MatrixField, MeshInstance

from taichi_extras.utils.mesh import element_field


def write(
    filepath: Path,
    points: np.ndarray,
    faces: np.ndarray,
    *,
    binary: bool = True,
    texture: Optional[str | np.ndarray] = None
) -> None:
    """
    Parameters:
        points : (n, 3)
        faces  : (m, 3) or (m, 4)
    """
    mesh: PolyData = pv.make_tri_mesh(points=points, faces=faces)
    mesh.save(filepath, binary=binary, texture=texture, recompute_normals=False)


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


def write_mesh(
    filepath: Path,
    mesh: MeshInstance,
    key: str = "position",
    *,
    binary: bool = True,
    texture: Optional[str | np.ndarray] = None
) -> None:
    position: np.ndarray
    if key and key in mesh.verts.keys:
        position_field: MatrixField = mesh.verts.get_member_field("position")
        position = position_field.to_numpy()
    else:
        position = mesh.get_position_as_numpy()
    indices: np.ndarray = get_faces(mesh=mesh)
    write(
        filepath=filepath,
        points=position,
        faces=indices,
        binary=binary,
        texture=texture,
    )
