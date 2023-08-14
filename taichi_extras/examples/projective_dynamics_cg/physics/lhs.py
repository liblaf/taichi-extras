import taichi as ti
from taichi import Matrix, MeshInstance, Vector

from taichi_extras.utils.mesh import element_field

from .const import MASS_DENSITY, SHEAR_MODULUS


@ti.kernel
def compute_hessian_kernel(
    mesh: ti.template(),  # type: ignore
    mass_density: float,
    shear_modulus: float,
):
    for c in mesh.cells:
        shape = Matrix.cols(
            [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
        )
        c.shape_undeformed_inverse = ti.math.inverse(shape)
        c.volume = ti.abs(ti.math.determinant(shape)) / 6.0
        for v in c.verts:
            v.mass += mass_density * c.volume / 4.0
        hessian = Matrix.zero(dt=float, n=4, m=4)
        H = (
            shear_modulus
            * c.volume
            * c.shape_undeformed_inverse  # type: ignore
            @ c.shape_undeformed_inverse.transpose()  # type: ignore
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


def compute_hessian(
    mesh: MeshInstance,
    *,
    mass_density: float = MASS_DENSITY,
    shear_modulus: float = SHEAR_MODULUS,
):
    element_field.place_safe(
        field=mesh.cells,
        members={
            "shape_undeformed_inverse": ti.math.mat3,
            "volume": float,
        },
    )
    element_field.place_safe(field=mesh.edges, members={"hessian": float})
    element_field.place_safe(
        field=mesh.verts,
        members={
            "hessian": float,
            "mass": float,
        },
    )
    compute_hessian_kernel(
        mesh=mesh,
        mass_density=mass_density,
        shear_modulus=shear_modulus,
    )
