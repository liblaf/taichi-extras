import typing
from pathlib import Path

import meshio
import numpy as np
import pyvista as pv


def _pyvista_faces_to_indices(faces: np.ndarray) -> np.ndarray:
    i: int = 0
    indices: list[np.ndarray] = list()
    while i < len(faces):
        assert faces[i] == 3
        indices.append(faces[i + 1 : i + faces[i] + 1])
        i += faces[i] + 1
    return np.array(indices)


def read_obj(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh: pv.PolyData = typing.cast(pv.PolyData, pv.read(filename=filepath))
    mesh: pv.PolyData = typing.cast(pv.PolyData, mesh.triangulate())
    return mesh.points, _pyvista_faces_to_indices(mesh.faces)


def write_obj(points: np.ndarray, faces: np.ndarray, filepath: str | Path) -> None:
    mesh = meshio.Mesh(points=points, cells=[("triangle", faces)])
    meshio.write(filename=filepath, mesh=mesh)
