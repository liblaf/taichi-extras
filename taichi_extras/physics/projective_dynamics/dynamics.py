import taichi as ti
from taichi import MatrixField, MeshInstance

from taichi_extras.lang.mesh import mesh_element_field

from . import conjugate_gradient, rhs
from .config import Constants


@ti.kernel
def compute_velocity_kernel(
    mesh: ti.template(),  # type: ignore
    time_step: float,
):
    for v in mesh.verts:
        v.velocity = (v.position - v.position_last) / time_step


def compute_velocity(
    mesh: MeshInstance,
    constants: Constants,
) -> None:
    compute_velocity_kernel(mesh=mesh, time_step=constants.time_step)


@ti.kernel
def update_position_kernel(mesh: ti.template()):  # type: ignore
    for v in mesh.verts:
        v.position += v.x


def update_position(mesh: MeshInstance) -> None:
    update_position_kernel(mesh=mesh)


def projective_dynamics(
    mesh: MeshInstance,
    constants: Constants,
    n_projective_dynamics_iter: int = 8,
    n_conjugate_gradient_iter: int = 1024,
) -> None:
    mesh_element_field.place(
        field=mesh.verts,
        members={
            "position_last": ti.math.vec3,
            "velocity": ti.math.vec3,
        },
    )
    position: MatrixField = mesh.verts.get_member_field("position")
    position_last: MatrixField = mesh.verts.get_member_field("position_last")
    rhs.compute_position_predict(mesh=mesh, constants=constants)
    position_last.copy_from(position)
    for _ in range(n_projective_dynamics_iter):
        rhs.compute_force(mesh=mesh, constants=constants)
        rhs.compute_b(mesh=mesh, constants=constants)
        conjugate_gradient.conjugate_gradient(
            mesh=mesh,
            constants=constants,
            n_iter=n_conjugate_gradient_iter,
        )
        update_position(mesh=mesh)
    compute_velocity(mesh=mesh, constants=constants)
