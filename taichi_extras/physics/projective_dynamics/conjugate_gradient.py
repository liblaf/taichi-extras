import logging

import taichi as ti
from taichi import MatrixField, MeshInstance

from taichi_extras.lang.mesh import mesh_element_field

from . import lhs
from .config import Constants


@ti.kernel
def a_add_b_mul_c_kernel(
    result: ti.template(),  # type: ignore
    a: ti.template(),  # type: ignore
    b: float,
    c: ti.template(),  # type: ignore
):
    for i in result:
        result[i] = a[i] + b * c[i]


def a_add_b_mul_c(
    result: MatrixField, a: MatrixField, b: float, c: MatrixField
) -> None:
    a_add_b_mul_c_kernel(result=result, a=a, b=b, c=c)


@ti.kernel
def dot_product_kernel(a: ti.template(), b: ti.template()) -> float:  # type: ignore
    result = 0.0
    for i in a:
        result += ti.math.dot(a[i], b[i])
    return result


def dot_product(a: MatrixField, b: MatrixField) -> float:
    return dot_product_kernel(a=a, b=b)


def conjugate_gradient(
    mesh: MeshInstance,
    *,
    constants: Constants,
    n_iter: int = 1024,
) -> MatrixField:
    mesh_element_field.place(
        field=mesh.verts,
        members={
            "A_matmul_p": ti.math.vec3,
            "b": ti.math.vec3,
            "p": ti.math.vec3,
            "r": ti.math.vec3,
            "x": ti.math.vec3,
        },
    )
    A_matmul_p: MatrixField = mesh.verts.get_member_field("A_matmul_p")
    b: MatrixField = mesh.verts.get_member_field("b")
    p: MatrixField = mesh.verts.get_member_field("p")
    r: MatrixField = mesh.verts.get_member_field("r")
    x: MatrixField = mesh.verts.get_member_field("x")
    x.fill(0.0)
    r.copy_from(b)
    p.copy_from(r)
    r_norm: float = dot_product(r, r)
    r_norm_init: float = r_norm
    for _ in range(n_iter):
        lhs.A_matmul_p(mesh=mesh)
        alpha: float = r_norm / dot_product(p, A_matmul_p)
        a_add_b_mul_c(result=x, a=x, b=alpha, c=p)
        a_add_b_mul_c(result=r, a=r, b=-alpha, c=A_matmul_p)
        r_norm_new: float = dot_product(r, r)
        if r_norm_new < constants.tolerance * r_norm_init:
            break
        beta: float = r_norm_new / r_norm
        r_norm = r_norm_new
        a_add_b_mul_c(result=p, a=r, b=beta, c=p)
    else:
        logging.warning("Conjugate Gradient did not converge")
    return x
