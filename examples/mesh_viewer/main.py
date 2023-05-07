import typing
from pathlib import Path

import numpy as np
import pyvista as pv
import taichi as ti
import tetgen
import typer

import taichi_extras.mesh.tet
import taichi_extras.utils.field.shape
import taichi_extras.utils.ui.window
from taichi_extras.patches.ui.camera import track_user_inputs

ti.init(kernel_profiler=True)


vertices: ti.MatrixField
surfaces: ti.MatrixField
indices: ti.ScalarField


app = typer.Typer(name="mesh_viewer")


def paint(scene: ti.ui.Scene, show_wireframe: bool = True):
    scene.mesh(vertices=vertices, indices=indices, color=(1, 0, 0), two_sided=True)
    if show_wireframe:
        scene.mesh(
            vertices=vertices,
            indices=indices,
            color=(0, 0, 0),
            two_sided=True,
            show_wireframe=show_wireframe,
        )


@app.command()
def main(
    filepath: Path = typer.Argument(None, exists=True),
    frame_interval: int = typer.Option(1, "-f", "--frame-interval"),
    max_frames: typing.Optional[int] = typer.Option(None, "--max-frames"),
    movement_speed: float = typer.Option(10.0, "-m", "--movement-speed"),
    off_screen: bool = typer.Option(False, "--off-screen"),
    output_prefix: Path = typer.Option(Path.cwd() / "frames", "-o", "--output-prefix"),
    output_stem: str = typer.Option("frame", "--output-stem"),
    pitch_speed: float = typer.Option(100.0, "-p", "--pitch-speed"),
    show_wireframe: bool = typer.Option(True, "-w", "--no-show-wireframe"),
    yaw_speed: float = typer.Option(100.0, "-y", "--yaw-speed"),
) -> None:
    global vertices, surfaces, indices
    if filepath:
        mesh: pv.PolyData = typing.cast(pv.PolyData, pv.read(filepath))
    else:
        mesh: pv.PolyData = typing.cast(pv.PolyData, pv.Sphere())
    mesh: pv.PolyData = typing.cast(pv.PolyData, mesh.triangulate())
    gen: tetgen.TetGen = tetgen.TetGen(mesh)
    gen.make_manifold(verbose=True)
    vertices_numpy, tetras_numpy = gen.tetrahedralize(verbose=2)
    surfaces_numpy = taichi_extras.mesh.tet.extract_surface(tetras_numpy)
    vertices = ti.Vector.field(n=3, dtype=float, shape=len(vertices_numpy))
    surfaces = ti.Vector.field(n=3, dtype=int, shape=len(surfaces_numpy))
    indices = ti.field(dtype=int, shape=3 * np.prod(surfaces.shape))

    vertices.from_numpy(vertices_numpy)
    surfaces.from_numpy(surfaces_numpy)
    taichi_extras.utils.field.shape.reshape(src=surfaces, new_shape=-1, output=indices)

    camera = ti.ui.Camera()
    scene = ti.ui.Scene()
    window = ti.ui.Window(
        name="Mesh Viewer", res=(1080, 1080), show_window=not off_screen
    )

    with taichi_extras.utils.ui.window.Window(window) as w:
        while w.next_frame(max_frames=max_frames):
            if off_screen:
                front = (camera.curr_lookat - camera.curr_position).normalized()
                camera.position(*(camera.curr_position - front * movement_speed / 30))
            else:
                track_user_inputs(
                    self=camera,
                    window=window,
                    movement_speed=movement_speed,
                    yaw_speed=yaw_speed,
                    pitch_speed=pitch_speed,
                    hold_key=ti.ui.ALT,
                )

            paint(scene=scene, show_wireframe=show_wireframe)
            scene.ambient_light([0.5, 0.5, 0.5])
            scene.point_light(
                pos=camera.curr_position,
                color=[0.5, 0.5, 0.5],
            )
            scene.set_camera(camera)
            canvas: ti.ui.Canvas = window.get_canvas()
            canvas.scene(scene)

            if off_screen:
                w.save_image(
                    interval=frame_interval, prefix=output_prefix, stem=output_stem
                )
            else:
                window.show()


if __name__ == "__main__":
    app()
