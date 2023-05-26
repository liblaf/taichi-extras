import taichi as ti

from .const import TIME_STEP


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
    result.fill(val=0.0)
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
