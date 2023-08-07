from pathlib import Path
from typing import Optional

import taichi as ti
from taichi import MatrixField, MeshInstance, ScalarNdarray, Vector
from taichi.linalg import SparseMatrix, SparseSolver

from taichi_extras.utils.mesh import element_field

from .const import FIXED_STIFFNESS, GRAVITY, MASS_DENSITY, SHEAR_MODULUS, TIME_STEP
from .fixed import init_fixed
from .lhs import compute_hessian, get_A
from .rhs import compute_b, compute_force, compute_position_predict, get_b


def init(
    mesh: MeshInstance,
    fixed_filepath: Optional[Path] = None,
    fixed_stiffness: float = FIXED_STIFFNESS,
    mass_density: float = MASS_DENSITY,
    shear_modulus: float = SHEAR_MODULUS,
    time_step: float = TIME_STEP,
) -> SparseSolver:
    element_field.place_safe(field=mesh.verts, members={"position": ti.math.vec3})
    position: MatrixField = mesh.verts.get_member_field("position")
    position.from_numpy(mesh.get_position_as_numpy())
    init_fixed(mesh=mesh, filepath=fixed_filepath)
    compute_hessian(
        mesh=mesh,
        mass_density=mass_density,
        shear_modulus=shear_modulus,
    )
    solver: SparseSolver = SparseSolver()
    A: SparseMatrix = get_A(
        mesh=mesh, fixed_stiffness=fixed_stiffness, time_step=time_step
    )
    solver.compute(A)
    return solver


@ti.kernel
def compute_velocity_kernel(
    mesh: ti.template(),  # type: ignore
    time_step: ti.f32,
):
    for v in mesh.verts:
        v.velocity = (v.position - v.position_previous) / time_step


def compute_velocity(
    mesh: MeshInstance,
    time_step: float = TIME_STEP,
) -> None:
    compute_velocity_kernel(mesh=mesh, time_step=time_step)


@ti.kernel
def update_position_kernel(
    mesh: ti.template(),  # type: ignore
    delta: ti.types.ndarray(),  # type: ignore
):
    for v in mesh.verts:
        for i in ti.static(range(3)):
            v.position[i] += delta[v.id * 3 + i]


def update_position(mesh: MeshInstance, delta: ScalarNdarray) -> None:
    update_position_kernel(mesh=mesh, delta=delta)


def projective_dynamics(
    mesh: MeshInstance,
    solver: SparseSolver,
    n_projective_dynamics_iter: int = 5,
    fixed_stiffness: float = FIXED_STIFFNESS,
    gravity: Vector = GRAVITY,
    shear_modulus: float = SHEAR_MODULUS,
    time_step: float = TIME_STEP,
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
        b: ScalarNdarray = get_b(mesh=mesh)
        delta: ScalarNdarray = solver.solve(b)
        update_position(mesh=mesh, delta=delta)
    compute_velocity(mesh=mesh, time_step=time_step)
