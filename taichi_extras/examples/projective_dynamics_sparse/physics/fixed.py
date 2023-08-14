from pathlib import Path
from typing import Optional

import numpy as np
import taichi as ti
from taichi import MatrixField, MeshInstance

from taichi_extras.io import node
from taichi_extras.utils.mesh import element_field


def init_fixed(mesh: MeshInstance, filepath: Optional[Path] = None) -> None:
    element_field.place_safe(field=mesh.verts, members={"fixed": ti.math.vec3})
    fixed_field: MatrixField = mesh.verts.get_member_field("fixed")
    if filepath:
        fixed_numpy: np.ndarray = node.read(filepath)
        fixed_field.from_numpy(fixed_numpy)
    else:
        fixed_field.fill(np.nan)
