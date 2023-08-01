from typing import cast

import numpy as np
import pyvista as pv


def get_vertices_indices(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    mesh = cast(pv.PolyData, mesh.triangulate())
    vertices: np.ndarray = mesh.points.copy()
    indices: np.ndarray = mesh.faces.reshape(-1, 4)[:, 1:]
    return vertices, indices
