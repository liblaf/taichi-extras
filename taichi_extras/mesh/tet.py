import itertools

import numpy as np

from taichi_extras.tetgen.const import INDICES


def extract_surface(tetras: np.ndarray) -> np.ndarray:
    surfaces: set[tuple[int, int, int]] = set()
    for vertices in tetras:
        # v1, v2, v3, v4 = vertices
        assert len(vertices) == 4
        for i in range(4):
            face: tuple[int, int, int] = tuple(
                vertices[INDICES[i][j]] for j in range(3)
            )
            # v1, v2, v3 = face
            is_inner: bool = False
            for p in itertools.permutations(face):
                if p in surfaces:
                    # inner face - shared by two tetrahedrons
                    surfaces.remove(p)
                    is_inner = True
            if not is_inner:
                surfaces.add(face)
    return np.array(list(surfaces))
