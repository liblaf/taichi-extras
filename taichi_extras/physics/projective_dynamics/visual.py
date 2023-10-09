from typing import cast

import numpy as np
import taichi as ti
from matplotlib.cm import ScalarMappable
from taichi import MatrixField, MeshInstance, ScalarField

from taichi_extras.lang.mesh import mesh_element_field


@ti.kernel
def compute_distance_kernel(mesh: ti.template()):  # type: ignore
    for v in mesh.verts:
        v.distance = ti.math.distance(v.position, v.position_init)


def compute_color(mesh: MeshInstance) -> MatrixField:
    mesh_element_field.place(
        field=mesh.verts,
        members={
            "color": ti.math.vec3,
            "distance": float,
        },
    )
    compute_distance_kernel(mesh=mesh)
    color_field: MatrixField = mesh.verts.get_member_field("color")
    color_preset_field: MatrixField = mesh.verts.get_member_field("color_preset")
    distance_field: ScalarField = mesh.verts.get_member_field("distance")
    color_preset_numpy: np.ndarray = color_preset_field.to_numpy()
    distance_numpy: np.ndarray = distance_field.to_numpy()
    mappable: ScalarMappable = ScalarMappable()
    rgba: np.ndarray = cast(np.ndarray, mappable.to_rgba(distance_numpy))
    rgb: np.ndarray = np.delete(rgba, 3, axis=-1)
    preset_indices: np.ndarray = np.apply_along_axis(
        lambda x: np.any(x), 1, color_preset_numpy
    )
    color_numpy: np.ndarray = color_preset_numpy.copy()
    color_numpy[~preset_indices] = rgb[~preset_indices]
    color_field.from_numpy(color_numpy)
    return color_field
