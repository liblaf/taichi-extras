from pathlib import Path

import numpy as np
import taichi as ti
import typer

from taichi_extras.mesh.obj import read_obj
from taichi_extras.optimize.minimize.gradient_descent.adam import Adam

ti.init()


num_points: int
num_faces: int
source_points: ti.MatrixField
target_points: ti.MatrixField
output_points: ti.MatrixField
faces: ti.MatrixField


params: ti.ScalarField = ti.field(dtype=float, shape=9, needs_grad=True)
transform: ti.MatrixField = ti.Matrix.field(
    n=3, m=3, dtype=float, shape=(), needs_grad=True
)
displacement: ti.MatrixField = ti.Vector.field(
    n=3, dtype=float, shape=(), needs_grad=True
)

loss: ti.ScalarField = ti.field(dtype=float, shape=(), needs_grad=True)


def init_fields(
    np_source_points: np.ndarray,
    np_target_points: np.ndarray,
    np_faces: np.ndarray,
) -> None:
    global num_points, num_faces, source_points, target_points, faces, output_points
    num_points = np_source_points.shape[0]
    num_faces = np_faces.shape[0]
    source_points = ti.Vector.field(n=3, dtype=float, shape=num_points)
    target_points = ti.Vector.field(n=3, dtype=float, shape=num_points)
    faces = ti.Vector.field(n=3, dtype=int, shape=num_faces)
    source_points.from_numpy(np_source_points)
    target_points.from_numpy(np_target_points)
    faces.from_numpy(np_faces)
    output_points = ti.Vector.field(n=3, dtype=float, shape=num_points, needs_grad=True)


@ti.func
def compute_shape(
    points: ti.template(), faces: ti.template(), face_idx: int
) -> ti.math.mat3:
    v1, v2, v3 = [points[i] for i in faces[face_idx]]
    cross = ti.math.cross(v2 - v1, v3 - v1)
    v4 = v1 + cross / ti.sqrt(ti.math.length(cross))
    return ti.Matrix.cols([v2 - v1, v3 - v1, v4 - v1])


@ti.kernel
def compute_transformation():
    transform[None] = ti.Matrix(
        arr=[
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], params[8]],
        ]
    )
    displacement[None] = ti.Vector(arr=[params[9], params[10], params[11]])


@ti.kernel
def compute_loss():
    for i in range(num_points):
        output_points[i] = transform[None] @ source_points[i] + displacement[None]
    for i in range(num_faces):
        output_shape = compute_shape(points=output_points, faces=faces, face_idx=i)
        target_shape = compute_shape(points=target_points, faces=faces, face_idx=i)
        loss[None] += (output_shape - target_shape).norm()


def loss_fn():
    compute_transformation()
    compute_loss()


def main(
    source_filepath: str = typer.Option(
        Path.cwd() / "data" / "s0.obj", "-s", "--source"
    ),
    target_filepath: str = typer.Option(
        Path.cwd() / "data" / "u0.obj", "-t", "--target"
    ),
    output_filepath: str = typer.Option(
        Path.cwd() / "data" / "su-transform.txt", "-o", "--output"
    ),
    iters: int = typer.Option(4096, "--iters"),
):
    source_points, source_faces = read_obj(source_filepath)
    target_points, target_faces = read_obj(target_filepath)
    init_fields(
        np_source_points=source_points,
        np_target_points=target_points,
        np_faces=source_faces,
    )

    params[0] = 1.0
    params[4] = 1.0
    params[8] = 1.0
    adam = Adam(loss_fn=loss_fn, loss=loss, x=(params,))

    try:
        adam.run(iters=iters)
    except KeyboardInterrupt:
        pass
    finally:
        compute_transformation()
        np.savetxt(output_filepath, np.reshape(params.to_numpy(), newshape=(-1,)))


if __name__ == "__main__":
    typer.run(main)
