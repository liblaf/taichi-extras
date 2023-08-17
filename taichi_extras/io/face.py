"""
https://www.wias-berlin.de/software/tetgen/fformats.face.html
"""

from pathlib import Path

import numpy as np

from .utils import minify


def read(filepath: Path, reverse: bool = True) -> np.ndarray:
    text: str = filepath.read_text()
    lines: list[str] = list(minify(text.splitlines()))

    # <# of faces> <boundary marker (0 or 1)>
    num_faces, boundary_marker = map(int, lines[0].split())
    faces: np.ndarray = np.zeros(shape=(num_faces, 3), dtype=np.int32)
    for line in lines[1 : num_faces + 1]:
        # <face #> <node> <node> <node> [boundary marker]
        index, *verts = map(int, line.split())
        if boundary_marker:
            del verts[-1]
        assert len(verts) == 3
        if reverse:
            faces[index] = verts[::-1]
        else:
            faces[index] = verts

    return faces
