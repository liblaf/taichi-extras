import taichi as ti

from . import conjugate_gradient
from . import math as math_utils
from .const import GRAVITY, MASS_DENSITY, SHEAR_MODULUS, TIME_STEP


def init(
    mesh: ti.MeshInstance,
    mass_density: float = MASS_DENSITY,
    shear_modulus: float = SHEAR_MODULUS,
):
    init_kernel(
        mesh=mesh,
        mass_density=mass_density,
        shear_modulus=shear_modulus,
    )


@ti.kernel
def init_kernel(mesh: ti.template(), mass_density: ti.f32, shear_modulus: ti.f32):
    for c in mesh.cells:
        shape = ti.Matrix.cols(
            [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
        )
        c.undeformed_shape_inverse = shape.inverse()
        c.volume = shape.determinant() / 6.0
        for i in range(4):
            c.verts[i].mass += mass_density * c.volume / 4.0

        hessian = ti.Matrix.zero(dt=ti.f32, n=4, m=4)
        H = (
            shear_modulus
            * c.volume
            * c.undeformed_shape_inverse
            @ c.undeformed_shape_inverse.transpose()
        )
        hessian = ti.Matrix.zero(dt=ti.f32, n=4, m=4)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                hessian[i, j] = H[i, j]
                hessian[i, 3] -= H[i, j]
                hessian[3, j] -= H[i, j]
                hessian[3, 3] += H[i, j]

        for z in ti.static(range(4)):
            v = c.verts[z]
            v.hessian += hessian[z, z]

        for e in c.edges:
            u = ti.Vector([0, 0])
            for i in ti.static(range(2)):
                for j in ti.static(range(4)):
                    if e.verts[i].id == c.verts[j].id:
                        u[i] = j
            e.hessian += hessian[u[0], u[1]]

    for v in mesh.verts:
        v.position *= 1.5


def projective_dynamics(
    mesh: ti.MeshInstance,
    shear_modulus: float = SHEAR_MODULUS,
    gravity: ti.Vector = GRAVITY,
    time_step: float = TIME_STEP,
    n_projective_dynamics_iter: int = 5,
    n_conjugate_gradient_iter: int = 30,
) -> None:
    position: ti.MatrixField = mesh.verts.get_member_field(key="position")
    position_predict: ti.MatrixField = mesh.verts.get_member_field(
        key="position_predict"
    )
    position_previous: ti.MatrixField = mesh.verts.get_member_field(
        key="position_previous"
    )
    velocity: ti.MatrixField = mesh.verts.get_member_field(key="velocity")

    # auxiliary variables for conjugate gradient method
    b: ti.MatrixField = mesh.verts.get_member_field(key="b")
    p: ti.MatrixField = mesh.verts.get_member_field(key="p")
    r: ti.MatrixField = mesh.verts.get_member_field(key="r")
    x: ti.MatrixField = mesh.verts.get_member_field(key="x")

    def A_mul_x(x: ti.MatrixField, time_step: float = TIME_STEP) -> ti.MatrixField:
        return conjugate_gradient.A_mul_x(mesh=mesh, x=x, time_step=time_step)

    position_previous.copy_from(position)
    for _ in range(n_projective_dynamics_iter):
        compute_force(mesh=mesh, shear_modulus=shear_modulus, gravity=gravity)
        math_utils.a_add_b_mul_c(
            result=position_predict, a=position, b=time_step, c=velocity
        )
        # conjugate gradient method
        # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
        conjugate_gradient.compute_b(mesh=mesh, time_step=time_step)
        x = conjugate_gradient.conjugate_gradient_method(
            b=b, p=p, r=r, x=x, A_mul_x=A_mul_x, n_iter=n_conjugate_gradient_iter
        )
        # q(k + 1) = q(k) + Delta
        math_utils.a_add_b_mul_c(position, position, 1.0, x)
    compute_velocity(mesh=mesh, time_step=time_step)
    apply_constraint_fixed(mesh=mesh)


def compute_force(
    mesh: ti.MeshInstance,
    shear_modulus: float = SHEAR_MODULUS,
    gravity: ti.Vector = GRAVITY,
) -> None:
    force: ti.MatrixField = mesh.verts.get_member_field(key="force")
    force.fill(val=0.0)
    compute_force_kernel(mesh=mesh, shear_modulus=shear_modulus, gravity=gravity)


@ti.kernel
def compute_force_kernel(
    mesh: ti.template(), shear_modulus: ti.f32, gravity: ti.math.vec3
):
    for c in mesh.cells:
        shape = ti.Matrix.cols(
            [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
        )
        deformation_gradient = shape @ c.undeformed_shape_inverse
        U, _, V = math_utils.positive_singular_value_decomposition_func(
            deformation_gradient
        )
        force = (
            shear_modulus
            * c.volume
            * (shape @ c.undeformed_shape_inverse - U @ V.transpose())
            @ c.undeformed_shape_inverse.transpose()
        )
        for i in ti.static(range(3)):
            c.verts[i].force += force[:, i]
            c.verts[3].force -= force[:, i]

    for v in mesh.verts:
        v.force += v.mass * gravity


def compute_velocity(mesh: ti.MeshInstance, time_step: float = TIME_STEP) -> None:
    compute_velocity_kernel(mesh=mesh, time_step=time_step)


@ti.kernel
def compute_velocity_kernel(mesh: ti.template(), time_step: ti.f32):
    for v in mesh.verts:
        v.velocity = (v.position - v.position_previous) / time_step


def apply_constraint_fixed(mesh: ti.MeshInstance) -> None:
    apply_constraint_fixed_kernel(mesh=mesh)


@ti.kernel
def apply_constraint_fixed_kernel(mesh: ti.template()):
    return
    for v in mesh.verts:
        for i in ti.static(range(3)):
            if not ti.math.isnan(v.fixed[i]):
                v.position[i] = v.fixed[i]
                v.velocity[i] = 0.0
