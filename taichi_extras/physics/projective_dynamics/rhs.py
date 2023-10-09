from typing import cast

import taichi as ti
from taichi import Matrix, MatrixField, MeshInstance, Vector

from taichi_extras.lang.mesh import mesh_element_field

from .config import Constants


@ti.kernel
def compute_position_predict_kernel(
    mesh: ti.template(),  # type: ignore
    time_step: float,
):
    for v in mesh.verts:
        v.position_predict = v.position + v.velocity * time_step


def compute_position_predict(mesh: MeshInstance, constants: Constants):
    mesh_element_field.place(
        field=mesh.verts, members={"position_predict": ti.math.vec3}
    )
    compute_position_predict_kernel(mesh=mesh, time_step=constants.time_step)


@ti.func
def singular_value_decomposition_func(
    mat3: ti.math.mat3,
) -> tuple[ti.math.mat3, ti.math.mat3, ti.math.mat3]:
    """
    mat3 = U @ Sigma @ V.transpose()
    """
    U, Sigma, V = ti.svd(mat3)
    if ti.math.determinant(U) < 0.0:
        U[:, 2] *= -1
        Sigma[2, 2] *= -1
    if ti.math.determinant(V) < 0.0:
        V[:, 2] *= -1
        Sigma[2, 2] *= -1
    return U, Sigma, V


@ti.kernel
def compute_force_kernel(
    mesh: ti.template(),  # type: ignore
    gravity: ti.math.vec3,
):
    for c in mesh.cells:
        shape = Matrix.cols(
            [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
        )
        deformation_gradient = shape @ c.shape_undeformed_inv
        U, Sigma, V = singular_value_decomposition_func(deformation_gradient)
        for i in ti.static(range(3)):
            Sigma[i, i] = ti.math.clamp(
                Sigma[i, i], c.strain_limit[0], c.strain_limit[1]
            )
        transformation = U @ V.transpose()
        force = (
            -c.stiffness
            * c.volume
            * (deformation_gradient - transformation)
            @ c.shape_undeformed_inv.transpose()
        )
        for i in ti.static(range(3)):
            c.verts[i].force += force[:, i]
            c.verts[3].force -= force[:, i]
    for v in mesh.verts:
        for i in ti.static(range(3)):
            v.force[i] += (
                v.mass * v.stiffness_fixed * (v.position_fixed[i] - v.position[i])
            )
    for v in mesh.verts:
        v.force += v.mass * gravity


def compute_force(mesh: MeshInstance, constants: Constants) -> None:
    mesh_element_field.place(field=mesh.verts, members={"force": ti.math.vec3})
    force: MatrixField = mesh.verts.get_member_field("force")
    force.fill(0.0)
    compute_force_kernel(mesh=mesh, gravity=Vector(constants.gravity))


@ti.kernel
def compute_b_kernel(
    mesh: ti.template(),  # type: ignore
    time_step: float,
):
    for v in mesh.verts:
        v.b = (
            -(1.0 / time_step**2) * v.mass * (v.position - v.position_predict)
            + v.force
        )


def compute_b(mesh: MeshInstance, constants: Constants) -> None:
    mesh_element_field.place(field=mesh.verts, members={"b": ti.math.vec3})
    compute_b_kernel(mesh=mesh, time_step=constants.time_step)
