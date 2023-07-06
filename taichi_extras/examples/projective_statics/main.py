from pathlib import Path
from typing import Optional

import numpy as np
import taichi as ti
import typer

import taichi_extras.examples.projective_statics.utils.math
import taichi_extras.examples.projective_statics.utils.projective_dynamics
import taichi_extras.patches.ui.camera
import taichi_extras.patches.ui.window
import taichi_extras.tetgen.io.results
from taichi_extras.examples.projective_statics.utils.const import (
    GRAVITY,
    MASS_DENSITY,
    SHEAR_MODULUS,
    TIME_STEP,
)


def init(mesh_filepath: Path) -> tuple[ti.MeshInstance, ti.ScalarField]:
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
            "position_predict": ti.math.vec3,  # position + TIME_STEP * velocity
            "position_previous": ti.math.vec3,
            "position": ti.math.vec3,
            "velocity": ti.math.vec3,
            "b": ti.math.vec3,  # auxiliary for conjugate gradient method
            "p": ti.math.vec3,  # auxiliary for conjugate gradient method
            "product": ti.math.vec3,  # auxiliary for conjugate gradient method
            "r": ti.math.vec3,  # auxiliary for conjugate gradient method
            "x": ti.math.vec3,  # auxiliary for conjugate gradient method
        }
    )
    fixed: ti.ScalarField = mesh.verts.get_member_field(key="fixed")
    fixed.from_numpy(np.loadtxt(fname=mesh_filepath.with_suffix(".fixed.txt")))
    position: ti.MatrixField = mesh.verts.get_member_field(key="position")
    position.from_numpy(mesh.get_position_as_numpy())
    indices: ti.ScalarField = ti.field(dtype=ti.i32, shape=faces.size)
    indices.from_numpy(faces.flatten())
    taichi_extras.examples.projective_statics.utils.projective_dynamics.init(
        mesh=mesh, mass_density=MASS_DENSITY, shear_modulus=SHEAR_MODULUS
    )
    return mesh, indices


def main(
    mesh_filepath: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
    ),
    max_frames: Optional[int] = typer.Option(None, "--max-frames"),
    off_screen: bool = typer.Option(False, "--off-screen"),
    profiler: bool = typer.Option(False, "--profiler"),
) -> None:
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

            taichi_extras.examples.projective_statics.utils.projective_dynamics.projective_dynamics(
                mesh=mesh,
                shear_modulus=SHEAR_MODULUS,
                gravity=GRAVITY,
                time_step=TIME_STEP,
            )

    if profiler:
        ti.profiler.print_kernel_profiler_info()
        ti.profiler.print_scoped_profiler_info()


if __name__ == "__main__":
    typer.run(main)
