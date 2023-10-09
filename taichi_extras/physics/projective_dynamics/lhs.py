import taichi as ti
from taichi import Matrix, MatrixField, MeshInstance, Vector

from taichi_extras.lang.mesh import mesh_element_field

from .config import Constants


@ti.kernel
def compute_A_kernel(mesh: ti.template(), time_step: float):  # type: ignore
    for c in mesh.cells:
        shA_matmul_pe = Matrix.cols(
            [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
        )
        c.shape_undeformed_inv = ti.math.inverse(shA_matmul_pe)
        c.volume = ti.abs(ti.math.determinant(shA_matmul_pe)) / 6.0
        for v in c.verts:
            v.mass += c.mass_density * c.volume / 4.0
        H = (
            c.stiffness
            * c.volume
            * c.shape_undeformed_inv
            @ c.shape_undeformed_inv.transpose()
        )
        hessian = Matrix.zero(dt=float, n=4, m=4)
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
            u = Vector([0, 0])
            for i in ti.static(range(2)):
                for j in ti.static(range(4)):
                    if e.verts[i].id == c.verts[j].id:
                        u[i] = j
            e.hessian += hessian[u[0], u[1]]
    for e in mesh.edges:
        e.A = e.hessian
    for v in mesh.verts:
        v.hessian += v.stiffness_fixed * v.mass
        v.A = v.mass / time_step**2 + v.hessian


def compute_A(mesh: MeshInstance, constants: Constants) -> None:
    mesh_element_field.place(
        field=mesh.cells,
        members={
            "shape_undeformed_inv": ti.math.mat3,
            "volume": float,
        },
    )
    mesh_element_field.place(
        field=mesh.edges,
        members={
            "A": float,
            "hessian": float,
        },
    )
    mesh_element_field.place(
        field=mesh.verts,
        members={
            "A": float,
            "hessian": float,
            "mass": float,
        },
    )
    compute_A_kernel(mesh=mesh, time_step=constants.time_step)


@ti.kernel
def A_matmul_p_kernel(mesh: ti.template()):  # type: ignore
    for e in mesh.edges:
        e.verts[0].A_matmul_p += e.A * e.verts[1].p
        e.verts[1].A_matmul_p += e.A * e.verts[0].p
    for v in mesh.verts:
        v.A_matmul_p += v.A * v.p


def A_matmul_p(mesh: MeshInstance) -> None:
    mesh_element_field.place(field=mesh.verts, members={"A_matmul_p": ti.math.vec3})
    A_matmul_p: MatrixField = mesh.verts.get_member_field("A_matmul_p")
    A_matmul_p.fill(0.0)
    A_matmul_p_kernel(mesh=mesh)
