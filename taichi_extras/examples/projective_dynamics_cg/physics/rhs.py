import taichi as ti
from taichi import Matrix, MatrixField, MeshInstance, Vector

from taichi_extras.utils.mesh import element_field

from .const import FIXED_STIFFNESS, GRAVITY, SHEAR_MODULUS, TIME_STEP
from .mathematics import positive_singular_value_decomposition_func


@ti.kernel
def compute_position_predict_kernel(
    mesh: ti.template(),  # type: ignore
    time_step: float,
):
    for v in mesh.verts:
        v.position_predict = v.position + time_step * v.velocity


def compute_position_predict(mesh: MeshInstance, *, time_step: float = TIME_STEP):
    element_field.place_safe(
        field=mesh.verts, members={"position_predict": ti.math.vec3}
    )
    compute_position_predict_kernel(mesh=mesh, time_step=time_step)


@ti.kernel
def compute_force_kernel(
    mesh: ti.template(),  # type: ignore
    fixed_stiffness: float,
    gravity: ti.math.vec3,
    shear_modulus: float,
):
    for c in mesh.cells:
        shape = Matrix.cols(
            [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
        )
        deformation_gradient = shape @ c.shape_undeformed_inverse
        U, Sigma, V = positive_singular_value_decomposition_func(deformation_gradient)
        force = (
            -shear_modulus
            * c.volume
            * (deformation_gradient - U @ V.transpose())
            @ c.shape_undeformed_inverse.transpose()
        )
        for i in ti.static(range(3)):
            c.verts[i].force += force[:, i]
            c.verts[3].force -= force[:, i]

    for v in mesh.verts:
        v.force += v.mass * gravity

    for v in mesh.verts:
        for i in ti.static(range(3)):
            if not ti.math.isnan(v.fixed[i]):
                v.force[i] += v.mass * fixed_stiffness * (v.fixed[i] - v.position[i])


def compute_force(
    mesh: MeshInstance,
    *,
    fixed_stiffness: float = FIXED_STIFFNESS,
    gravity: Vector = GRAVITY,
    shear_modulus: float = SHEAR_MODULUS,
) -> None:
    element_field.place_safe(field=mesh.verts, members={"force": ti.math.vec3})
    force: MatrixField = mesh.verts.get_member_field("force")
    force.fill(0.0)
    compute_force_kernel(
        mesh=mesh,
        fixed_stiffness=fixed_stiffness,
        gravity=gravity,
        shear_modulus=shear_modulus,
    )


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


def compute_b(mesh: MeshInstance, time_step: float = TIME_STEP) -> None:
    element_field.place_safe(field=mesh.verts, members={"b": ti.math.vec3})
    compute_b_kernel(mesh=mesh, time_step=time_step)
