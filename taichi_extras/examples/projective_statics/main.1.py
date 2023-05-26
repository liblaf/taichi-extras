from pathlib import Path
from typing import Optional

import numpy as np
import taichi as ti
import typer

import taichi_extras.math.mat3
import taichi_extras.patches.ui.camera
import taichi_extras.patches.ui.window
import taichi_extras.tetgen.io.results

E, nu = 1e4, 0.0
GRAVITY: ti.Vector = ti.Vector([0.0, 0.0, 0.0])
MASS_DENSITY: float = 1000.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
TIME_STEP: float = 1e-1


mesh: ti.MeshInstance


def init(mesh_filepath: Path) -> tuple[ti.MeshInstance, ti.ScalarField]:
    global mesh
    mesh, faces = taichi_extras.tetgen.io.results.read(
        mesh_filepath,
        relations=["CE", "CV", "EV"],
    )
    mesh.cells.place(
        members={
            "undeformed_shape_inverse": ti.math.mat3,
            "volume": ti.f32,
        }
    )
    mesh.edges.place(
        members={
            "hessian": ti.f32,
        }
    )
    mesh.verts.place(
        members={
            "fixed": ti.math.vec3,
            "force": ti.math.vec3,
            "hessian": ti.f32,
            "mass": ti.f32,
            "position": ti.math.vec3,
            "velocity": ti.math.vec3,
            "b": ti.math.vec3,  # auxiliary
            "dx": ti.math.vec3,  # auxiliary
            "multiply_by_hessian": ti.math.vec3,  # auxiliary
            "p0": ti.math.vec3,  # auxiliary
            "r0": ti.math.vec3,  # auxiliary
            "x0": ti.math.vec3,  # auxiliary
            "y": ti.math.vec3,  # auxiliary
        }
    )
    fixed: ti.ScalarField = mesh.verts.get_member_field(key="fixed")
    fixed.from_numpy(np.loadtxt(fname=mesh_filepath.with_suffix(".fixed.txt")))
    position: ti.MatrixField = mesh.verts.get_member_field(key="position")
    position.from_numpy(mesh.get_position_as_numpy())
    indices: ti.ScalarField = ti.field(dtype=ti.i32, shape=faces.size)
    indices.from_numpy(faces.flatten())
    compute_init()
    return mesh, indices


@ti.kernel
def compute_init_kernel():
    for c in mesh.cells:
        shape = ti.Matrix.cols(
            [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
        )
        c.undeformed_shape_inverse = shape.inverse()
        c.volume = shape.determinant() / 6.0
        for i in range(4):
            c.verts[i].mass = MASS_DENSITY * c.volume / 4.0

        hessian = ti.Matrix.zero(dt=ti.f32, n=4, m=4)
        for u in range(4):
            dD = ti.Matrix.zero(ti.f32, 3, 3)
            if u == 3:
                for j in ti.static(range(3)):
                    dD[0, j] = -1
            else:
                dD[0, u] = 1
            dF = dD @ c.undeformed_shape_inverse
            dP = 2.0 * mu * dF
            dH = -c.volume * dP @ c.undeformed_shape_inverse.transpose()
            for i in ti.static(range(3)):
                for j in ti.static(range(1)):
                    hessian[i, u] -= (TIME_STEP**2) * dH[j, i]
                    hessian[3, u] += (TIME_STEP**2) * dH[j, i]

        for e in c.edges:
            u = ti.Vector([0, 0])
            for i in ti.static(range(2)):
                for j in ti.static(range(4)):
                    if e.verts[i].id == c.verts[j].id:
                        u[i] = j
            e.hessian += hessian[u[0], u[1]]

        for z in range(c.verts.size):
            v = c.verts[z]
            v.hessian += hessian[z, z]


def compute_init() -> None:
    compute_init_kernel()


@ti.kernel
def element_add(result: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
    for i in result:
        result[i] = a[i] + k * b[i]


@ti.kernel
def conditional_element_add(
    result: ti.template(),
    a: ti.template(),
    k: ti.f32,
    b: ti.template(),
    condition: ti.template(),
):
    for i in result:
        if ti.math.isnan(condition[i]).any():
            result[i] = a[i] + k * b[i]
        else:
            result[i] = condition[i]


@ti.kernel
def sum_element_dot(a: ti.template(), b: ti.template()) -> ti.f32:
    result = 0.0
    for i in a:
        result += a[i].dot(b[i])
    return result


@ti.kernel
def compute_velocity_kernel():
    for v in mesh.verts:
        v.velocity = (v.position - v.x0) / TIME_STEP


def compute_velocity() -> None:
    compute_velocity_kernel()


@ti.kernel
def compute_force_kernel():
    for c in mesh.cells:
        shape = ti.Matrix.cols(
            [c.verts[i].position - c.verts[3].position for i in ti.static(range(3))]
        )
        deformation_gradient = shape @ c.undeformed_shape_inverse
        U, Sigma, V = taichi_extras.math.mat3.positive_singular_value_decomposition(
            deformation_gradient
        )
        P = 2 * mu * (deformation_gradient - U @ V.transpose())
        H = -c.volume * P @ c.undeformed_shape_inverse.transpose()
        for i in ti.static(range(3)):
            Hx = ti.Vector([H[j, i] for j in ti.static(range(3))])
            c.verts[i].force += Hx
            c.verts[3].force -= Hx
    for v in mesh.verts:
        v.force += v.mass * GRAVITY


def compute_force() -> None:
    force: ti.MatrixField = mesh.verts.get_member_field(key="force")
    force.fill(val=0.0)
    compute_force_kernel()


@ti.kernel
def multiply_by_hessian_kernel(result: ti.template(), position: ti.template()):
    for v in mesh.verts:
        result[v.id] = v.mass * position[v.id] + v.hessian * position[v.id]
    for e in mesh.edges:
        u, v = e.verts[0].id, e.verts[1].id
        result[u] += e.hessian * position[v]
        result[v] += e.hessian * position[u]


def multiply_by_hessian(position) -> ti.MatrixField:
    result: ti.MatrixField = mesh.verts.get_member_field(key="multiply_by_hessian")
    multiply_by_hessian_kernel(result=result, position=position)
    return result


@ti.kernel
def compute_b_kernel():
    for v in mesh.verts:
        v.b = v.mass * (v.position - v.y) - (TIME_STEP**2) * v.force


def compute_residual() -> None:
    compute_b_kernel()


def projective_dynamics() -> None:
    b: ti.MatrixField = mesh.verts.get_member_field(key="b")
    dx: ti.MatrixField = mesh.verts.get_member_field(key="dx")
    fixed: ti.ScalarField = mesh.verts.get_member_field(key="fixed")
    p0: ti.MatrixField = mesh.verts.get_member_field(key="p0")
    position: ti.MatrixField = mesh.verts.get_member_field(key="position")
    r0: ti.MatrixField = mesh.verts.get_member_field(key="r0")
    velocity: ti.MatrixField = mesh.verts.get_member_field(key="velocity")
    x0: ti.MatrixField = mesh.verts.get_member_field(key="x0")
    y: ti.MatrixField = mesh.verts.get_member_field(key="y")
    x0.copy_from(position)
    element_add(result=y, a=position, k=TIME_STEP, b=velocity)
    for _ in range(5):  # Projective Dynamics iterations
        compute_force()
        compute_residual()
        dx.fill(val=0.0)
        r0.copy_from(b)
        p0.copy_from(r0)
        residual_square: float = sum_element_dot(b, b)
        residual_square_init: float = residual_square
        residual_square_new: float = residual_square
        acceptable_residual_square: float = residual_square_init * (1e-6**2)
        for _ in range(30):  # Conjugate Gradient Method iterations
            q: ti.MatrixField = multiply_by_hessian(p0)
            alpha: float = residual_square_new / sum_element_dot(p0, q)
            element_add(dx, dx, alpha, p0)
            element_add(r0, r0, -alpha, q)
            residual_square: float = residual_square_new
            residual_square_new: float = sum_element_dot(r0, r0)
            if residual_square_new <= acceptable_residual_square:
                break
            beta: float = residual_square_new / residual_square
            element_add(p0, r0, beta, p0)
        # element_add(position, position, -1.0, dx)
        conditional_element_add(position, position, -1.0, dx, condition=fixed)
    compute_velocity()


def main(
    mesh_filepath: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
    ),
    max_frames: Optional[int] = typer.Option(None, "--max-frames"),
    off_screen: bool = typer.Option(False, "--off-screen"),
    profiler: bool = typer.Option(False, "--profiler"),
) -> None:
    global mesh

    ti.init(kernel_profiler=profiler)

    mesh, indices = init(mesh_filepath=mesh_filepath)

    camera: ti.ui.Camera = ti.ui.Camera()
    scene: ti.ui.Scene = ti.ui.Scene()
    window: ti.ui.Window = ti.ui.Window(
        name="Projective Statics", res=(800, 600), show_window=not off_screen
    )
    canvas = window.get_canvas()
    camera.lookat(x=0.0, y=0.0, z=0.0)
    camera.position(x=0.0, y=0.0, z=4.0)

    with taichi_extras.patches.ui.window.Window(
        window=window, frame_interval=1 if off_screen else None
    ) as w:
        while w.next_frame(max_frames=max_frames):
            if not off_screen:
                taichi_extras.patches.ui.camera.track_user_inputs(
                    self=camera, window=window
                )
            scene.set_camera(camera=camera)
            scene.ambient_light(color=(0.8, 0.8, 0.8))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))

            scene.mesh(
                vertices=mesh.verts.get_member_field(key="position"),
                indices=indices,
                show_wireframe=True,
            )

            canvas.scene(scene=scene)
            if not off_screen:
                window.show()

            projective_dynamics()

    if profiler:
        ti.profiler.print_kernel_profiler_info()
        ti.profiler.print_scoped_profiler_info()


if __name__ == "__main__":
    typer.run(main)
