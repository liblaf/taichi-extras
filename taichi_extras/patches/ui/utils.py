import math

import taichi as ti


def euler_to_vec(yaw: float, pitch: float) -> ti.Vector:
    v = ti.Vector([0.0, 0.0, 0.0])
    v[0] = math.sin(yaw) * math.cos(pitch)
    v[1] = math.sin(pitch)
    v[2] = math.cos(yaw) * math.cos(pitch)
    return v


def vec_to_euler(v) -> tuple[float, float]:
    v = v.normalized()
    pitch: float = math.asin(v[1])

    sin_yaw: float = v[0] / math.cos(pitch)
    cos_yaw: float = v[2] / math.cos(pitch)
    cos_yaw = max(-1.0, min(1.0, cos_yaw))
    yaw = math.acos(cos_yaw)
    if sin_yaw < 0:
        yaw = -yaw

    return yaw, pitch
