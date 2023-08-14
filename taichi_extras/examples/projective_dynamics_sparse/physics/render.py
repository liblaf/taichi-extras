from typing import cast

import matplotlib.cm
import numpy as np
import taichi as ti
from matplotlib.colors import Colormap
from taichi import MatrixField, MeshInstance, ScalarField

from taichi_extras.utils.mesh import element_field


@ti.kernel
def compute_color_kernel(mesh: ti.template()):  # type: ignore
    for v in mesh.verts:
        v.displacement = ti.math.length(v.position - v.position_init)


def compute_color(mesh: MeshInstance) -> MatrixField:
    element_field.place_safe(
        field=mesh.verts, members={"color": ti.math.vec3, "displacement": float}
    )
    compute_color_kernel(mesh=mesh)
    color: MatrixField = mesh.verts.get_member_field("color")
    displacement: ScalarField = mesh.verts.get_member_field("displacement")
    displacement_numpy: np.ndarray = displacement.to_numpy()
    displacement_numpy /= displacement_numpy.max()
    colormap: Colormap = matplotlib.cm.get_cmap("plasma")
    color_numpy: np.ndarray = cast(np.ndarray, colormap(displacement_numpy))
    color_numpy = color_numpy[:, :3]
    color.from_numpy(color_numpy)
    return color
