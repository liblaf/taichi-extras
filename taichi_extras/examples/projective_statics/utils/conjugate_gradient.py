from typing import Callable

import numpy as np
import taichi as ti

from .const import TIME_STEP, TOLERANCE
from .math import a_add_b_mul_c, dot


def conjugate_gradient_method(
    b: ti.MatrixField,
    p: ti.MatrixField,
    r: ti.MatrixField,
    x: ti.MatrixField,
    A_mul_x: Callable,
    n_iter: int = 30,
    relative_tolerance: float = TOLERANCE**2,
    absolute_tolerance: float = TOLERANCE**2,
) -> ti.MatrixField:
    """
    https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    """
    # x_0 = 0
    x.fill(val=0.0)
    # r_0 = b - A @ x_0
    a_add_b_mul_c(r, b, -1.0, A_mul_x(x))
    r_square: float = dot(r, r)
    if np.allclose(r_square, 0.0, rtol=relative_tolerance, atol=absolute_tolerance):
        return x
    # p_0 = r_0
    p.copy_from(r)
    for _ in range(n_iter):
        r_squared: float = dot(r, r)  # r_k^T @ r_k
        # alpha = r_k^T @ r_k / (p_k^T @ A @ p_k)
        alpha: float = r_squared / dot(p, A_mul_x(p))
        # x_{k + 1} = x_k + alpha * p_k
        a_add_b_mul_c(x, x, alpha, p)
        # r_{k + 1} = r_k - alpha * A @ p_k
        a_add_b_mul_c(r, r, -alpha, A_mul_x(p))
        r_new_squared: float = dot(r, r)
        if r_new_squared < relative_tolerance * r_square:
            break
        # beta = r_{k + 1}^T @ r_{k + 1} / r_k^T @ r_k
        beta: float = r_new_squared / r_squared
        # p_{k + 1} = r_{k + 1} + beta * p_k
        a_add_b_mul_c(p, r, beta, p)
    return x


def compute_b(mesh: ti.MeshInstance, time_step: float = TIME_STEP) -> None:
    compute_b_kernel(mesh=mesh, time_step=time_step)


@ti.kernel
def compute_b_kernel(mesh: ti.template(), time_step: float):
    for v in mesh.verts:
        v.b = (
            -(1.0 / time_step**2) * v.mass * (v.position - v.position_predict)
            + v.force
        )


def A_mul_x(
    mesh: ti.MeshInstance,
    x: ti.MatrixField,
    time_step: float = TIME_STEP,
    result_key: str = "product",
) -> ti.MatrixField:
    result: ti.MatrixField = mesh.verts.get_member_field(key=result_key)
    A_mul_x_kernel(mesh=mesh, result=result, x=x, time_step=time_step)
    return result


@ti.kernel
def A_mul_x_kernel(
    mesh: ti.template(), result: ti.template(), x: ti.template(), time_step: ti.f32
):
    for v in mesh.verts:
        result[v.id] = (1.0 / time_step**2) * v.mass * x[v.id] + v.hessian * x[v.id]
    for e in mesh.edges:
        u, v = e.verts[0], e.verts[1]
        result[u.id] += e.hessian * x[v.id]
        result[v.id] += e.hessian * x[u.id]
