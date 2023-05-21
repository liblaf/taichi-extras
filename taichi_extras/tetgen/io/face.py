from pathlib import Path

import numpy as np


def read(filepath: str | Path, reverse: bool = True) -> np.ndarray:
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    text: str = filepath.read_text()
    lines: list[str] = text.splitlines()
    lines: list[str] = list(map(str.strip, lines))
    lines: list[str] = list(filter(lambda line: not line.startswith("#"), lines))

    num_faces, boundary_marker = map(int, lines[0].split())
    faces: np.ndarray = np.zeros(shape=(num_faces, 3), dtype=np.int32)
    for line in lines[1:]:
        index, *verts = map(int, line.split())
        if boundary_marker:
            assert verts[-1] == -1
            del verts[-1]
        assert len(verts) == 3
        if reverse:
            faces[index] = verts[::-1]
        else:
            faces[index] = verts

    return faces
