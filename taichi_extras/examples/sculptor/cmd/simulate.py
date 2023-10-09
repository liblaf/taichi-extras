import logging
import time
from pathlib import Path
from typing import Annotated, Optional

import taichi as ti
import typer
from taichi import MatrixField, ScalarField
from taichi.ui import Canvas, Scene

from taichi_extras.io import node, tetgen
from taichi_extras.io.tetgen import TetMesh
from taichi_extras.physics.projective_dynamics import dynamics, init, visual
from taichi_extras.physics.projective_dynamics.config import Config
from taichi_extras.typer.run import run as typer_run
from taichi_extras.ui.camera import Camera
from taichi_extras.ui.window import Window

ti.init()


def main(
    mesh: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    *,
    max_frames: Annotated[int, typer.Option()] = 90,
    output: Annotated[
        Path, typer.Option(exists=False, dir_okay=False, writable=True, readable=False)
    ],
    show_window: Annotated[bool, typer.Option()] = False,
    video_dir: Annotated[
        Optional[Path],
        typer.Option(exists=True, file_okay=False, writable=True, readable=False),
    ] = None,
) -> None:
    time_start: float = time.perf_counter()
    config_model: Config = Config.load_yaml(config)
    tet_mesh: TetMesh = tetgen.read_all(mesh, relations=["CE", "CV", "EV"])
    camera: Camera = Camera()
    scene: Scene = Scene()
    window: Window = Window(
        name="Projective Dynamics", output_dir=video_dir, show_window=show_window
    )
    canvas: Canvas = window.get_canvas()
    camera.lookat(*config_model.camera.lookat)
    camera.position(*config_model.camera.position)

    tet_mesh.instance.verts.place({"position": ti.math.vec3})
    position: MatrixField = tet_mesh.instance.verts.get_member_field("position")
    position.from_numpy(tet_mesh.instance.get_position_as_numpy())
    indices: ScalarField = ti.field(dtype=ti.i32, shape=(tet_mesh.faces.size,))
    indices.from_numpy(tet_mesh.faces.flatten())

    init.init(mesh=tet_mesh, config=config_model)

    with window:
        while window.next_frame(max_frames=max_frames, track_user_input=camera):
            if video_dir or show_window:
                color: MatrixField = visual.compute_color(mesh=tet_mesh.instance)
                scene.mesh(
                    vertices=position,
                    indices=indices,
                    per_vertex_color=color,
                    two_sided=True,
                    show_wireframe=True,
                )
                scene.set_camera(camera=camera)
                scene.ambient_light(color=(1.0, 1.0, 1.0))
                canvas.scene(scene=scene)
            dynamics.projective_dynamics(
                mesh=tet_mesh.instance,
                constants=config_model.constants,
                n_projective_dynamics_iter=8,
                n_conjugate_gradient_iter=1024,
            )
    time_end: float = time.perf_counter()
    logging.info(f"Elapsed Time: {time_end - time_start} s")
    node.write(output, points=position.to_numpy())


if __name__ == "__main__":
    typer_run(main)
