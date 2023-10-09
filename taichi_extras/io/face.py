"""
https://www.wias-berlin.de/software/tetgen/fformats.face.html
"""

from pathlib import Path

import numpy as np

from .utils import minify


def read(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    text: str = filepath.read_text()
    lines: list[str] = list(minify(text.splitlines()))
    # <# of faces> <boundary marker (0 or 1)>
    num_faces, boundary_marker = map(int, lines[0].split())
    faces: np.ndarray = np.zeros(shape=(num_faces, 3), dtype=int)
    boundary_markers: np.ndarray = np.zeros(shape=(num_faces,), dtype=int)
    for line in lines[1 : num_faces + 1]:
        # <face #> <node> <node> <node> [boundary marker]
        words: list[str] = line.split()
        index: int = int(words[0])
        faces[index] = list(map(int, words[1:4]))
        if boundary_marker:
            boundary_markers[index] = int(words[4])
    return faces, boundary_markers
