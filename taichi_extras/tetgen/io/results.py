from pathlib import Path

import meshtaichi_patcher
import numpy as np
import taichi as ti

import taichi_extras.tetgen.io.face


def read(
    filepath: str | Path, relations: list[str] = ["CV"]
) -> tuple[ti.MeshInstance, np.ndarray]:
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    mesh: ti.MeshInstance = meshtaichi_patcher.load_mesh(
        meshes=str(filepath), relations=relations
    )

    face_filepath: Path = filepath.with_suffix(".face")
    faces = taichi_extras.tetgen.io.face.read(face_filepath)

    return mesh, faces
