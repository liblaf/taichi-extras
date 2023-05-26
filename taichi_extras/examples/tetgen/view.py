from pathlib import Path

import taichi as ti
import typer

import taichi_extras.patches.ui.camera
import taichi_extras.patches.ui.window
import taichi_extras.tetgen.const
import taichi_extras.tetgen.io.results

ti.init()


def extract_faces(mesh: ti.MeshInstance) -> ti.ScalarField:
    indices: ti.ScalarField = ti.field(dtype=ti.i32, shape=len(mesh.faces) * 3)

    @ti.kernel
    def kernel():
        INDICES = ti.static(taichi_extras.tetgen.const.INDICES)
        for c in mesh.cells:
            center = ti.Vector.zero(dt=ti.f32, n=3)
            for v in c.verts:
                center += v.position
            center /= 4.0
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    if center.y < 0.0:
                        indices[(c.id * 4 + i) * 3 + j] = c.verts[INDICES[i][j]].id
                    else:
                        indices[(c.id * 4 + i) * 3 + j] = 0

    kernel()

    return indices


def main(
    mesh_filepath: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
    )
) -> None:
    mesh, faces = taichi_extras.tetgen.io.results.read(mesh_filepath)

    mesh.verts.place({"position": ti.math.vec3})
    position: ti.MatrixField = mesh.verts.get_member_field(key="position")
    position.from_numpy(mesh.get_position_as_numpy())

    indices: ti.ScalarField = ti.field(dtype=ti.i32, shape=faces.size)
    indices.from_numpy(faces.flatten())

    tet_indices: ti.ScalarField = extract_faces(mesh=mesh)

    window: ti.ui.Window = ti.ui.Window(name="TetGen", res=(800, 600))
    camera: ti.ui.Camera = ti.ui.Camera()
    scene: ti.ui.Scene = ti.ui.Scene()
    canvas: ti.ui.Canvas = window.get_canvas()

    camera.lookat(x=0.0, y=0.0, z=0.0)
    camera.position(x=0.0, y=1.0, z=2.0)

    with taichi_extras.patches.ui.window.Window(
        window=window, frame_interval=None
    ) as w:
        while w.next_frame():
            taichi_extras.patches.ui.camera.track_user_inputs(
                self=camera, window=window
            )
            scene.set_camera(camera=camera)
            scene.ambient_light(color=(0.8, 0.8, 0.8))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))

            # scene.mesh(
            #     vertices=position,
            #     indices=indices,
            #     color=(1.0, 0.0, 0.0),
            #     show_wireframe=True,
            # )
            scene.mesh(vertices=position, indices=tet_indices)
            scene.mesh(
                vertices=position,
                indices=tet_indices,
                color=(0.0, 0.0, 0.0),
                show_wireframe=True,
            )

            canvas.scene(scene=scene)
            window.show()


if __name__ == "__main__":
    typer.run(main)
