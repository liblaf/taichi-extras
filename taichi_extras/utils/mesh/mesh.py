import numpy as np
from taichi import MatrixField, MeshInstance


def get_bounding_box(mesh: MeshInstance, key: str = "position") -> np.ndarray:
    position: MatrixField = mesh.verts.get_member_field(key)
    position_numpy: np.ndarray = position.to_numpy()
    return np.array(
        [
            np.min(position_numpy, axis=0),
            np.max(position_numpy, axis=0),
        ]
    )
