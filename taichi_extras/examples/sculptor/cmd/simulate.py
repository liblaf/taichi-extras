import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import taichi as ti
import typer
from pyvista import PolyData
from taichi import MatrixField, MeshInstance
from taichi.ui import Canvas, Scene

from taichi_extras.io import node
from taichi_extras.io import pyvista as io_pv
from taichi_extras.physics.projective_dynamics import dynamics, fixed, render
from taichi_extras.physics.projective_dynamics.const import (
    FIXED_STIFFNESS,
    GRAVITY,
    MASS_DENSITY,
    SHEAR_MODULUS,
    TIME_STEP,
    Method,
)
from taichi_extras.ui.camera import Camera
from taichi_extras.ui.window import Window

ti.init()


def init_fixed(
    mesh: MeshInstance, topology: np.ndarray, inner: Sequence[Optional[PolyData]]
) -> None:
    assert topology.shape == (1 + len(inner),)
    position: np.ndarray = np.nan * np.ones(shape=(len(mesh.verts), 3))
    for i, inner_mesh in enumerate(inner):
        if not inner_mesh:
            continue
        begin: int = topology[i]
        end: int = topology[i + 1]
        position[begin:end] = inner_mesh.points
    fixed.init_fixed(mesh=mesh, position=position)


def main(
    mesh: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    topology: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    inner: Annotated[
        list[Path],
        typer.Option(
            exists=False, file_okay=True, dir_okay=False, writable=False, readable=True
        ),
    ],
    *,
    max_frames: Annotated[int, typer.Option()] = 90,
    output: Annotated[
        Path,
        typer.Option(
            exists=False, file_okay=True, dir_okay=False, writable=True, readable=False
        ),
    ],
    show_window: Annotated[bool, typer.Option()] = False,
    video_dir: Annotated[
        Optional[Path],
        typer.Option(
            exists=True, file_okay=False, dir_okay=True, writable=True, readable=False
        ),
    ] = None,
) -> None:
    time_start: float = time.perf_counter()
    mesh_instance, faces = node.read_all(mesh, relations=["CE", "CV", "EV", "FV"])
    dynamics.init(
        mesh=mesh_instance,
        fixed_stiffness=FIXED_STIFFNESS,
        mass_density=MASS_DENSITY,
        shear_modulus=SHEAR_MODULUS,
        time_step=TIME_STEP,
        method=Method.CG,
    )
    init_fixed(
        mesh=mesh_instance,
        topology=np.loadtxt(topology, dtype=int),
        inner=list(
            map(
                lambda filepath: None
                if filepath.name == "EMPTY"
                else io_pv.read_poly_data(filepath),
                inner,
            )
        ),
    )

    camera: Camera = Camera()
    scene: Scene = Scene()
    window: Window = Window(
        name="Projective Dynamics", output_dir=video_dir, show_window=show_window
    )
    canvas: Canvas = window.get_canvas()
    camera.position(0.0, 0.0, 0.26)
    camera.lookat(0.0, 0.0, 0.0)

    position: ti.MatrixField = mesh_instance.verts.get_member_field("position")
    indices: ti.ScalarField = ti.field(dtype=ti.i32, shape=(faces.size,))
    indices.from_numpy(faces.flatten())

    with window:
        while window.next_frame(max_frames=max_frames, track_user_input=camera):
            if video_dir or show_window:
                color: MatrixField = render.compute_color(mesh=mesh_instance)
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

            dynamics.projective_dynamics(
                mesh=mesh_instance,
                fixed_stiffness=FIXED_STIFFNESS,
                gravity=GRAVITY,
                shear_modulus=SHEAR_MODULUS,
                time_step=TIME_STEP,
            )
    time_end: float = time.perf_counter()
    logging.info(f"Elapsed Time: {time_end - time_start} s")
    node.write(output, position=position.to_numpy())


if __name__ == "__main__":
    typer.run(main)
