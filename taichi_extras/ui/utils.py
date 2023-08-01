import math
from typing import cast

import taichi as ti


def clamp(x, xmin, xmax):
    return max(xmin, min(xmax, x))


def euler_to_vec(yaw: float, pitch: float) -> ti.Vector:
    v = ti.Vector([0.0, 0.0, 0.0])
    v[0] = math.sin(yaw) * math.cos(pitch)
    v[1] = math.sin(pitch)
    v[2] = math.cos(yaw) * math.cos(pitch)
    return v


def vec_to_euler(v: ti.Vector) -> tuple[float, float]:
    v = cast(ti.Vector, v.normalized())
    pitch: float = math.asin(v[1])

    sin_yaw: float = v[0] / math.cos(pitch)  # type: ignore
    cos_yaw: float = v[2] / math.cos(pitch)  # type: ignore
    cos_yaw = clamp(x=cos_yaw, xmin=-1.0, xmax=+1.0)
    yaw = math.acos(cos_yaw)
    if sin_yaw < 0:
        yaw = -yaw

    return yaw, pitch
