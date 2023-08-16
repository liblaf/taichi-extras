import logging
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import taichi as ti
import typer
from physics.const import (
    FIXED_STIFFNESS,
    GRAVITY,
    MASS_DENSITY,
    SHEAR_MODULUS,
    TIME_STEP,
    Method,
)
from physics.dynamics import init, projective_dynamics
from physics.render import compute_color
from taichi import MatrixField
from taichi.linalg import SparseSolver
from taichi.ui.canvas import Canvas
from taichi.ui.scene import Scene

from taichi_extras.io import node, pyvista
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
    camera_position: Annotated[
        tuple[float, float, float], typer.Option("-c", "--camera")
    ] = (0.0, 0.0, 3.0),
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
    max_frames: Annotated[int, typer.Option("-f", "--max-frames")] = sys.maxsize,
    method: Annotated[
        Method,
        typer.Option("-m", "--method", case_sensitive=False),
    ] = Method.CG,
    output: Annotated[Path, typer.Option("-o", "--output")] = Path.cwd() / "output",
    record: Annotated[bool, typer.Option("--record/--no-record")] = True,
    show_window: Annotated[bool, typer.Option("-w", "--show-window")] = False,
    fixed_stiffness: Annotated[
        float, typer.Option("--fixed-stiffness")
    ] = FIXED_STIFFNESS,
) -> None:
    time_start: float = time.perf_counter()
    mesh, faces = node.read_all(input, relations=["CE", "CV", "EV", "FV"])
    solver: Optional[SparseSolver] = init(
        mesh=mesh,
        fixed_filepath=fixed,
        fixed_stiffness=fixed_stiffness,
        mass_density=MASS_DENSITY,
        shear_modulus=SHEAR_MODULUS,
        time_step=TIME_STEP,
        method=method,
    )

    camera: Camera = Camera()
    scene: Scene = Scene()
    window: Window = Window(
        name="Projective Dynamics",
        output_dir=output,
        record=record,
        show_window=show_window,
    )
    canvas: Canvas = window.get_canvas()
    camera.position(*camera_position)
    camera.lookat(0.0, 0.0, 0.0)

    position: ti.MatrixField = mesh.verts.get_member_field("position")
    indices: ti.ScalarField = ti.field(dtype=ti.i32, shape=(faces.size,))
    indices.from_numpy(faces.flatten())

    color: MatrixField
    with window:
        while window.next_frame(max_frames=max_frames, track_user_input=camera):
            if record or show_window:
                color = compute_color(mesh=mesh)
                scene.mesh(
                    vertices=position,
                    indices=indices,
                    per_vertex_color=color,
                    show_wireframe=True,
                )
                scene.set_camera(camera=camera)
                scene.ambient_light(color=(0.5, 0.5, 0.5))
                scene.point_light(pos=(2.0, 2.0, 2.0), color=(2.0, 2.0, 2.0))
                canvas.scene(scene=scene)

            projective_dynamics(
                mesh=mesh,
                solver=solver,
                fixed_stiffness=fixed_stiffness,
                gravity=GRAVITY,
                shear_modulus=SHEAR_MODULUS,
                time_step=TIME_STEP,
            )
    time_end: float = time.perf_counter()
    logging.info(f"Elapsed Time: {time_end - time_start} s")
    color = compute_color(mesh=mesh)
    texture: np.ndarray = np.asarray(color.to_numpy() * 255.0, dtype=np.uint8)
    pyvista.write(
        output / "tri.ply",
        points=position.to_numpy(),
        faces=faces,
        binary=False,
        texture=texture,
    )
    pyvista.write_mesh(output / "tet.ply", mesh=mesh, binary=False, texture=texture)


if __name__ == "__main__":
    typer.run(main)
