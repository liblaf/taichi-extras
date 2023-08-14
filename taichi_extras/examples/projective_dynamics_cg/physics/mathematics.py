import taichi as ti
from taichi import MatrixField, MeshInstance

from taichi_extras.utils.mesh import element_field

from .const import FIXED_STIFFNESS, TIME_STEP


@ti.func
def positive_singular_value_decomposition_func(
    mat3: ti.math.mat3,
) -> tuple[ti.math.mat3, ti.math.mat3, ti.math.mat3]:
    """
    mat3 = U @ Sigma @ V.transpose()
    """
    U, Sigma, V = ti.svd(mat3)
    if ti.math.determinant(U) < 0.0:  # type: ignore
        U[:, 2] *= -1
        Sigma[2, 2] *= -1
    if ti.math.determinant(V) < 0.0:  # type: ignore
        V[:, 2] *= -1
        Sigma[2, 2] *= -1
    return U, Sigma, V


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
def compute_Ap_kernel(
    mesh: ti.template(),  # type: ignore
    fixed_stiffness: float,
    time_step: float,
):
    for v in mesh.verts:
        v.Ap += (v.mass / (time_step**2) + v.hessian) * v.p
        for i in ti.static(range(3)):
            if not ti.math.isnan(v.fixed[i]):
                v.Ap += fixed_stiffness * v.mass * v.p

    for e in mesh.edges:
        e.verts[0].Ap += e.hessian * e.verts[1].p
        e.verts[1].Ap += e.hessian * e.verts[0].p


def compute_Ap(
    mesh: MeshInstance,
    fixed_stiffness: float = FIXED_STIFFNESS,
    time_step: float = TIME_STEP,
) -> None:
    Ap: MatrixField = mesh.verts.get_member_field("Ap")
    Ap.fill(0.0)
    compute_Ap_kernel(mesh=mesh, fixed_stiffness=fixed_stiffness, time_step=time_step)


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
    fixed_stiffness: float = FIXED_STIFFNESS,
    time_step: float = TIME_STEP,
    n_iter: int = 30,
) -> MatrixField:
    element_field.place_safe(
        field=mesh.verts,
        members={
            "Ap": ti.math.vec3,
            "p": ti.math.vec3,
            "r": ti.math.vec3,
            "x": ti.math.vec3,
        },
    )
    Ap: MatrixField = mesh.verts.get_member_field("Ap")
    b: MatrixField = mesh.verts.get_member_field("b")
    p: MatrixField = mesh.verts.get_member_field("p")
    r: MatrixField = mesh.verts.get_member_field("r")
    x: MatrixField = mesh.verts.get_member_field("x")
    x.fill(0.0)
    r.copy_from(b)
    p.copy_from(r)
    r_norm: float = dot_product(r, r)
    for _ in range(n_iter):
        compute_Ap(mesh=mesh, fixed_stiffness=fixed_stiffness, time_step=time_step)
        alpha: float = r_norm / dot_product(p, Ap)
        a_add_b_mul_c(result=x, a=x, b=alpha, c=p)
        a_add_b_mul_c(result=r, a=r, b=-alpha, c=Ap)
        r_norm_new: float = dot_product(r, r)
        beta: float = r_norm_new / r_norm
        r_norm = r_norm_new
        a_add_b_mul_c(result=p, a=r, b=beta, c=p)
    return x
