import sys
from pathlib import Path
from typing import Annotated, Optional

import taichi as ti
import typer
from physics.const import (
    FIXED_STIFFNESS,
    GRAVITY,
    MASS_DENSITY,
    SHEAR_MODULUS,
    TIME_STEP,
)
from taichi.linalg import SparseSolver
from taichi.ui.canvas import Canvas
from taichi.ui.scene import Scene

from taichi_extras.examples.projective_dynamics_hard.physics.dynamics import (
    init,
    projective_dynamics,
)
from taichi_extras.examples.projective_dynamics_hard.physics.fixed import init_fixed
from taichi_extras.io import node, trimesh
from taichi_extras.ui.camera import Camera
from taichi_extras.ui.window import Window

ti.init()


def main(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, readable=True, writable=False
        ),
    ],
    *,
    fixed: Annotated[
        Optional[Path],
        typer.Option(
            "-f",
            "--fixed",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            writable=False,
        ),
    ] = None,
    frame_interval: Annotated[int, typer.Option("-i", "--frame-interval")] = 1,
    max_frames: Annotated[int, typer.Option("-f", "--max-frames")] = sys.maxsize,
    output: Annotated[Path, typer.Option("-o", "--output")] = Path.cwd() / "output",
    show_window: Annotated[bool, typer.Option("-w", "--show-window")] = False
) -> None:
    mesh, faces = node.read_all(input, relations=["CE", "CV", "EV", "FV"])
    solver: SparseSolver = init(
        mesh=mesh,
        fixed_filepath=fixed,
        fixed_stiffness=FIXED_STIFFNESS,
        mass_density=MASS_DENSITY,
        shear_modulus=SHEAR_MODULUS,
        time_step=TIME_STEP,
    )

    camera: Camera = Camera()
    scene: Scene = Scene()
    window: Window = Window(
        name="Projective Dynamics",
        res=(1280, 960),
        show_window=show_window,
        frame_interval=frame_interval,
        output_dir=output,
    )
    canvas: Canvas = window.get_canvas()
    camera.position(0.0, 0.0, 3.0)
    camera.lookat(0.0, 0.0, 0.0)

    position: ti.MatrixField = mesh.verts.get_member_field("position")
    indices: ti.ScalarField = ti.field(dtype=ti.i32, shape=(faces.size,))
    indices.from_numpy(faces.flatten())

    with window:
        while window.next_frame(max_frames=max_frames, track_user_input=camera):
            if show_window or frame_interval > 0:
                scene.mesh(
                    vertices=position,
                    indices=indices,
                    show_wireframe=True,
                )
                scene.set_camera(camera=camera)
                scene.ambient_light(color=(0.5, 0.5, 0.5))
                scene.point_light(pos=(2.0, 2.0, 2.0), color=(2.0, 2.0, 2.0))
                canvas.scene(scene=scene)

            projective_dynamics(
                mesh=mesh,
                solver=solver,
                fixed_stiffness=FIXED_STIFFNESS,
                gravity=GRAVITY,
                shear_modulus=SHEAR_MODULUS,
                time_step=TIME_STEP,
            )

    trimesh.write(output / "tri.off", vertices=position.to_numpy(), faces=faces)
    trimesh.write_mesh(output / "tet.off", mesh=mesh)


if __name__ == "__main__":
    typer.run(main)
