from typing import Optional

import numpy as np
import taichi as ti
from taichi import MatrixField, MeshInstance

from taichi_extras.utils.mesh import element_field


def init_fixed(mesh: MeshInstance, position: Optional[np.ndarray] = None) -> None:
    element_field.place_safe(field=mesh.verts, members={"fixed": ti.math.vec3})
    fixed_field: MatrixField = mesh.verts.get_member_field("fixed")
    if position is not None:
        fixed_field.from_numpy(position)
    else:
        fixed_field.fill(np.nan)
