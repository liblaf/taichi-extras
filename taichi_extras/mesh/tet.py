import itertools

import numpy as np


def extract_surface(tetras: np.ndarray) -> np.ndarray:
    surfaces: set[tuple[int, int, int]] = set()
    for vertices in tetras:
        # v1, v2, v3, v4 = vertices
        assert len(vertices) == 4
        for face in itertools.combinations(vertices, 3):
            # v1, v2, v3 = face
            for p in itertools.permutations(face):
                if p
            if face in surfaces:
                # inner face - shared by two tetrahedrons
                surfaces.remove(face)
            else:  # outer face
                surfaces.add(tuple(sorted(face)))
    return np.array(list(surfaces))
