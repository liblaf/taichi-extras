import taichi as ti

from . import conjugate_gradient
from .const import GRAVITY, MASS_DENSITY, SHEAR_MODULUS, TIME_STEP
from .math import a_add_b_mul_c, dot, positive_singular_value_decomposition_func


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

    position_previous.copy_from(position)
    for _ in range(n_projective_dynamics_iter):
        compute_force(mesh=mesh, shear_modulus=shear_modulus, gravity=gravity)
        a_add_b_mul_c(result=position_predict, a=position, b=time_step, c=velocity)
        # conjugate gradient method
        # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
        conjugate_gradient.compute_b(mesh=mesh, time_step=time_step)
        # x_0 = 0
        x.fill(val=0.0)
        # r_0 = b - A @ x_0
        a_add_b_mul_c(
            r, b, -1.0, conjugate_gradient.A_mul_x(mesh, x, time_step=time_step)
        )
        r_squared_init: float = dot(r, r)
        r_squared_allow: float = r_squared_init * 1e-10
        if r_squared_init > r_squared_allow:
            # p_0 = r_0
            p.copy_from(r)
            for _ in range(n_conjugate_gradient_iter):
                r_squared: float = dot(r, r)  # r_k^T @ r_k
                # alpha = r_k^T @ r_k / (p_k^T @ A @ p_k)
                alpha: float = r_squared / dot(
                    p, conjugate_gradient.A_mul_x(mesh, p, time_step=time_step)
                )
                # x_{k + 1} = x_k + alpha * p_k
                a_add_b_mul_c(x, x, alpha, p)
                # r_{k + 1} = r_k - alpha * A @ p_k
                a_add_b_mul_c(
                    r,
                    r,
                    -alpha,
                    conjugate_gradient.A_mul_x(mesh, p, time_step=time_step),
                )
                r_new_squared: float = dot(r, r)
                if r_new_squared < r_squared_allow:
                    break
                # beta = r_{k + 1}^T @ r_{k + 1} / r_k^T @ r_k
                beta: float = r_new_squared / r_squared
                print(r_squared, r_new_squared, beta)
                # p_{k + 1} = r_{k + 1} + beta * p_k
                a_add_b_mul_c(p, r, beta, p)
        # q(k + 1) = q(k) + Delta
        a_add_b_mul_c(position, position, 1.0, x)
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
        U, _, V = positive_singular_value_decomposition_func(deformation_gradient)
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
