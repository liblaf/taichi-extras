from pathlib import Path

import meshtaichi_patcher
import numpy as np
import taichi as ti

from . import face


def read(filepath: Path, relations: list[str] = ["CV"]) -> ti.MeshInstance:
    mesh: ti.MeshInstance = meshtaichi_patcher.load_mesh(
        meshes=str(filepath), relations=relations
    )
    return mesh


def read_all(
    filepath: Path, relations: list[str] = ["CV"]
) -> tuple[ti.MeshInstance, np.ndarray]:
    mesh: ti.MeshInstance = read(filepath, relations=relations)
    face_filepath: Path = filepath.with_suffix(".face")
    faces = face.read(face_filepath)
    return mesh, faces
