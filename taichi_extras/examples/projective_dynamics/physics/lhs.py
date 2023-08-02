import taichi as ti
from taichi import Matrix, MatrixField, MeshInstance, Vector
from taichi.linalg import SparseMatrix, SparseMatrixBuilder

from taichi_extras.utils.mesh import element_field

from .const import MASS_DENSITY, SHEAR_MODULUS, TIME_STEP


@ti.kernel
def compute_hessian_kernel(
    mesh: ti.template(),  # type: ignore
    mass_density: ti.f32,
    shear_modulus: ti.f32,
):
    for c in mesh.cells:
        shape = Matrix.cols(
            [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
        )
        c.shape_undeformed_inverse = shape.inverse()
        c.volume = ti.abs(shape.determinant()) / 6.0  # type: ignore
        # for vv in range(c.verts.size):
        #     v = c.verts[vv]
        for v in c.verts:
            v.mass += mass_density * c.volume / 4.0
        hessian = Matrix.zero(dt=ti.f32, n=4, m=4)
        H = (
            shear_modulus
            * c.volume
            * c.shape_undeformed_inverse
            @ c.shape_undeformed_inverse.transpose()  # type: ignore
        )
        hessian = Matrix.zero(dt=ti.f32, n=4, m=4)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                hessian[i, j] = H[i, j]
                hessian[i, 3] -= H[i, j]
                hessian[3, j] -= H[i, j]
                hessian[3, 3] += H[i, j]

        for z in ti.static(range(4)):
            v = c.verts[z]
            v.hessian += hessian[z, z]

        # for ee in range(c.edges.size):
        #     e = c.edges[ee]
        for e in c.edges:
            u = Vector([0, 0])
            for i in ti.static(range(2)):
                for j in ti.static(range(4)):
                    if e.verts[i].id == c.verts[j].id:
                        u[i] = j
            e.hessian += hessian[u[0], u[1]]


def compute_hessian(
    mesh: MeshInstance,
    mass_density: float = MASS_DENSITY,
    shear_modulus: float = SHEAR_MODULUS,
):
    element_field.place_safe(
        field=mesh.cells,
        members={
            "shape_undeformed_inverse": ti.math.mat3,
            "volume": ti.f32,
        },
    )
    element_field.place_safe(field=mesh.edges, members={"hessian": ti.f32})
    element_field.place_safe(
        field=mesh.verts,
        members={
            "hessian": ti.f32,
            "mass": ti.f32,
            "position": ti.math.vec3,
        },
    )
    position: MatrixField = mesh.verts.get_member_field("position")
    position.from_numpy(mesh.get_position_as_numpy())
    compute_hessian_kernel(
        mesh=mesh,
        mass_density=mass_density,
        shear_modulus=shear_modulus,
    )


@ti.kernel
def get_A_kernel(
    mesh: ti.template(),  # type: ignore
    builder: ti.types.sparse_matrix_builder(),  # type: ignore
    time_step: ti.f32,
):
    for v in mesh.verts:
        for i in ti.static(range(3)):
            builder[v.id * 3 + i, v.id * 3 + i] += v.mass / (time_step**2) + v.hessian

    for e in mesh.edges:
        for i in ti.static(range(3)):
            builder[e.verts[0].id * 3 + i, e.verts[1].id * 3 + i] += e.hessian
            builder[e.verts[1].id * 3 + i, e.verts[0].id * 3 + i] += e.hessian


def get_A(mesh: MeshInstance, time_step: float = TIME_STEP) -> SparseMatrix:
    builder: SparseMatrixBuilder = SparseMatrixBuilder(
        num_rows=len(mesh.verts) * 3, num_cols=len(mesh.verts) * 3
    )
    get_A_kernel(mesh=mesh, builder=builder, time_step=time_step)
    position: MatrixField = mesh.verts.get_member_field("position")
    print(position.to_numpy())
    return builder.build()
