"""
https://www.wias-berlin.de/software/tetgen/fformats.node.html
"""

from pathlib import Path

import meshtaichi_patcher
import numpy as np
import taichi as ti

from . import face
from .utils import minify


def read(filepath: Path) -> np.ndarray:
    text: str = filepath.read_text()
    lines: list[str] = list(minify(text.splitlines()))

    # <# of points> <dimension (must be 3)> <# of attributes> <# of boundary markers (0 or 1)>
    num_points, dimension, num_attributes, boundary = map(int, lines[0].split())
    assert dimension == 3
    assert num_attributes == 0
    points: np.ndarray = np.zeros(shape=(num_points, 3), dtype=np.float32)
    for line in lines[1:]:
        # <point #> <x> <y> <z> [attributes] [boundary marker]
        index: int = int(line.split()[0])
        position: list[float] = list(map(float, line.split()[1:]))
        if boundary:
            del position[-1]
        assert len(position) == 3
        points[index] = position

    return points


def read_mesh(filepath: Path, relations: list[str] = ["CV"]) -> ti.MeshInstance:
    mesh: ti.MeshInstance = meshtaichi_patcher.load_mesh(
        meshes=str(filepath), relations=relations
    )
    return mesh


def read_all(
    filepath: Path, relations: list[str] = ["CV"]
) -> tuple[ti.MeshInstance, np.ndarray]:
    mesh: ti.MeshInstance = read_mesh(filepath, relations=relations)
    face_filepath: Path = filepath.with_suffix(".face")
    faces = face.read(face_filepath)
    return mesh, faces


def write(filepath: Path, position: np.ndarray, *, fmt: str = ".18e") -> None:
    assert position.ndim == 2
    assert position.shape[1] == 3
    with filepath.open(mode="w") as fp:
        # <# of points> <dimension (must be 3)> <# of attributes> <# of boundary markers (0 or 1)>
        print(position.shape[0], position.shape[1], 0, 0, file=fp)
        for i, p in enumerate(position):
            # <point #> <x> <y> <z> [attributes] [boundary marker]
            print(i, *map(lambda x: format(x, fmt), p), file=fp)
