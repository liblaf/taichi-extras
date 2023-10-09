from typing import cast

import numpy as np
from pyvista import PolyData


def find_enclosed_point(surface: PolyData, max_iter: int = 1000) -> np.ndarray:
    surface = cast(PolyData, surface.compute_normals(auto_orient_normals=True))
    point: np.ndarray = surface.points[0]
    normal: np.ndarray = surface.point_normals[0]
    offset: float = 1.0
    for _ in range(max_iter):
        points: PolyData = PolyData(point - offset * normal)
        enclosed: PolyData = points.select_enclosed_points(surface)
        if enclosed.point_data["SelectedPoints"].any():
            return enclosed.points[0]
        offset /= 2.0
    raise RuntimeError("Failed to find inner point")


def get_points_faces(mesh: PolyData) -> tuple[np.ndarray, np.ndarray]:
    points: np.ndarray = mesh.points.copy()
    faces: np.ndarray = mesh.faces.reshape(-1, 4)[:, 1:].copy()
    return points, faces


def make_mesh(points: np.ndarray, faces: np.ndarray) -> PolyData:
    pv_faces: np.ndarray = np.zeros(
        shape=(faces.shape[0], faces.shape[1] + 1), dtype=faces.dtype
    )
    pv_faces[:, 0] = faces.shape[1]
    pv_faces[:, 1:] = faces
    mesh: PolyData = PolyData(points, faces=pv_faces)
    return mesh
