from pathlib import Path
from typing import Optional

import taichi as ti
from taichi import MatrixField, MeshInstance, Vector

from taichi_extras.utils.mesh import element_field

from .const import FIXED_STIFFNESS, GRAVITY, MASS_DENSITY, SHEAR_MODULUS, TIME_STEP
from .fixed import init_fixed
from .lhs import compute_hessian
from .mathematics import conjugate_gradient
from .rhs import compute_b, compute_force, compute_position_predict


def init(
    mesh: MeshInstance,
    fixed_filepath: Optional[Path] = None,
    *,
    mass_density: float = MASS_DENSITY,
    shear_modulus: float = SHEAR_MODULUS,
) -> None:
    element_field.place_safe(field=mesh.verts, members={"position": ti.math.vec3})
    position: MatrixField = mesh.verts.get_member_field("position")
    position.from_numpy(mesh.get_position_as_numpy())
    init_fixed(mesh=mesh, filepath=fixed_filepath)
    compute_hessian(
        mesh=mesh,
        mass_density=mass_density,
        shear_modulus=shear_modulus,
    )


@ti.kernel
def compute_velocity_kernel(
    mesh: ti.template(),  # type: ignore
    time_step: float,
):
    for v in mesh.verts:
        v.velocity = (v.position - v.position_previous) / time_step


def compute_velocity(
    mesh: MeshInstance,
    *,
    time_step: float = TIME_STEP,
) -> None:
    compute_velocity_kernel(mesh=mesh, time_step=time_step)


@ti.kernel
def update_position_kernel(mesh: ti.template()):  # type: ignore
    for v in mesh.verts:
        v.position += v.x


def update_position(mesh: MeshInstance) -> None:
    update_position_kernel(mesh=mesh)


def projective_dynamics(
    mesh: MeshInstance,
    *,
    fixed_stiffness: float = FIXED_STIFFNESS,
    gravity: Vector = GRAVITY,
    shear_modulus: float = SHEAR_MODULUS,
    time_step: float = TIME_STEP,
    n_conjugate_gradient_iter: int = 30,
    n_projective_dynamics_iter: int = 5,
) -> None:
    element_field.place_safe(
        field=mesh.verts,
        members={"position_previous": ti.math.vec3, "velocity": ti.math.vec3},
    )
    position: MatrixField = mesh.verts.get_member_field("position")
    position_previous: MatrixField = mesh.verts.get_member_field("position_previous")
    compute_position_predict(mesh=mesh, time_step=time_step)
    position_previous.copy_from(position)

    for _ in range(n_projective_dynamics_iter):
        compute_force(
            mesh=mesh,
            fixed_stiffness=fixed_stiffness,
            gravity=gravity,
            shear_modulus=shear_modulus,
        )
        compute_b(mesh=mesh, time_step=time_step)
        conjugate_gradient(
            mesh=mesh,
            fixed_stiffness=fixed_stiffness,
            time_step=time_step,
            n_iter=n_conjugate_gradient_iter,
        )
        update_position(mesh=mesh)
    compute_velocity(mesh=mesh, time_step=time_step)
