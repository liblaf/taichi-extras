import dataclasses
from pathlib import Path

import meshtaichi_patcher
import numpy as np
from taichi import MeshInstance

from . import ele, face, node


@dataclasses.dataclass(kw_only=True)
class TetMesh:
    instance: MeshInstance
    vert_attrs: np.ndarray
    point_boundary_markers: np.ndarray
    faces: np.ndarray
    face_boundary_markers: np.ndarray
    cell_attrs: np.ndarray


def read_mesh(filepath: Path, relations: list[str] = ["CV"]) -> MeshInstance:
    mesh: MeshInstance = meshtaichi_patcher.load_mesh(
        meshes=str(filepath), relations=relations
    )
    return mesh


def read_all(filepath: Path, relations: list[str] = ["CV"]) -> TetMesh:
    instance: MeshInstance = read_mesh(filepath, relations=relations)
    ele_filepath: Path = filepath.with_suffix(".ele")
    face_filepath: Path = filepath.with_suffix(".face")
    node_filepath: Path = filepath.with_suffix(".node")
    tets, cell_attrs = ele.read(ele_filepath)
    faces, face_boundary_markers = face.read(face_filepath)
    points, vert_attrs, point_boundary_markers = node.read(node_filepath)
    return TetMesh(
        instance=instance,
        vert_attrs=vert_attrs,
        point_boundary_markers=point_boundary_markers,
        faces=faces,
        face_boundary_markers=face_boundary_markers,
        cell_attrs=cell_attrs,
    )
