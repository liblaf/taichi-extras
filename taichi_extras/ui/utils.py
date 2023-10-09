from typing import cast

import taichi as ti
from taichi import Vector


def clamp(x: float, xmin: float, xmax: float) -> float:
    return max(xmin, min(xmax, x))


def euler_to_vec(yaw: float, pitch: float) -> Vector:
    v = Vector([0.0, 0.0, 0.0])
    v[0] = ti.sin(yaw) * ti.cos(pitch)
    v[1] = ti.sin(pitch)
    v[2] = ti.cos(yaw) * ti.cos(pitch)
    return v


def vec_to_euler(v: Vector) -> tuple[float, float]:
    v = cast(Vector, v.normalized())
    pitch: float = ti.asin(v[1])
    sin_yaw: float = cast(float, v[0] / ti.cos(pitch))
    cos_yaw: float = cast(float, v[2] / ti.cos(pitch))
    cos_yaw = clamp(cos_yaw, xmin=-1.0, xmax=+1.0)
    yaw: float = ti.acos(cos_yaw)
    if sin_yaw < 0:
        yaw = -yaw
    return yaw, pitch
