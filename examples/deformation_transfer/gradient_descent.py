import re
from pathlib import Path

import numpy as np
import taichi as ti
import typer

from taichi_extras.mesh.obj import read_obj, write_obj
from taichi_extras.optimize.minimize.gradient_descent.adam import Adam

ti.init()


num_points: int
num_faces: int
source_points: ti.MatrixField
target_points: ti.MatrixField
faces: ti.MatrixField
input_points: ti.MatrixField
output_points: ti.MatrixField


loss: ti.ScalarField = ti.field(dtype=float, shape=(), needs_grad=True)


def init_fields(
    np_source_points: np.ndarray,
    np_target_points: np.ndarray,
    np_faces: np.ndarray,
    np_input_points: np.ndarray,
) -> None:
    global num_points, num_faces, source_points, target_points, faces, input_points, output_points
    num_points = np_source_points.shape[0]
    num_faces = np_faces.shape[0]
    source_points = ti.Vector.field(n=3, dtype=float, shape=num_points)
    target_points = ti.Vector.field(n=3, dtype=float, shape=num_points)
    faces = ti.Vector.field(n=3, dtype=int, shape=num_faces)
    input_points = ti.Vector.field(n=3, dtype=float, shape=num_points)
    output_points = ti.Vector.field(n=3, dtype=float, shape=num_points, needs_grad=True)
    source_points.from_numpy(np_source_points)
    target_points.from_numpy(np_target_points)
    faces.from_numpy(np_faces)
    input_points.from_numpy(np_input_points)


@ti.func
def compute_shape(
    points: ti.template(),  # type: ignore
    faces: ti.template(),  # type: ignore
    face_idx: int,
) -> ti.Matrix:
    v1, v2, v3 = [points[i] for i in faces[face_idx]]
    cross = (v2 - v1).cross(v3 - v1)
    v4 = v1 + cross / ti.sqrt(cross.norm())
    return ti.Matrix.cols([v2 - v1, v3 - v1, v4 - v1])


@ti.func
def compute_transformation(
    points: ti.template(),  # type: ignore
    points_deformed: ti.template(),  # type: ignore
    faces: ti.template(),  # type: ignore
    face_idx: int,
) -> ti.Matrix:
    shape = compute_shape(points=points, faces=faces, face_idx=face_idx)
    shape_deformed = compute_shape(
        points=points_deformed, faces=faces, face_idx=face_idx
    )
    return shape_deformed @ shape.inverse()  # type: ignore


@ti.kernel
def compute_loss():
    for i in range(num_faces):
        source_transformation = compute_transformation(
            points=source_points, points_deformed=input_points, faces=faces, face_idx=i
        )
        target_transformation = compute_transformation(
            points=target_points, points_deformed=output_points, faces=faces, face_idx=i
        )
        loss[None] += (source_transformation - target_transformation).norm()


def main(
    source_reference_filepath: str = typer.Option(
        Path.cwd() / "data" / "s0.obj", "-s", "--source-reference"
    ),
    target_reference_filepath: str = typer.Option(
        Path.cwd() / "data" / "t0.obj", "-t", "--target-reference"
    ),
    input_filepath: str = typer.Option(Path.cwd() / "data" / "s1.obj", "-i", "--input"),
    output_filepath: str = typer.Option(
        Path.cwd() / "data" / "t1.obj", "-o", "--output"
    ),
    iters: int = typer.Option(65536, "--iters"),
    beta_1: float = typer.Option(0.9),
    beta_2: float = typer.Option(0.999),
    epsilon: float = typer.Option(1e-8),
    eta: float = typer.Option(1e-3),
    initial_guess_filepath: str = typer.Option(None, "--initial-guess"),
):
    source_points, source_faces = read_obj(source_reference_filepath)
    target_points, target_faces = read_obj(target_reference_filepath)
    input_points, input_faces = read_obj(input_filepath)
    init_fields(
        np_source_points=source_points,
        np_target_points=target_points,
        np_faces=source_faces,
        np_input_points=input_points,
    )
    if initial_guess_filepath is None:
        iter_start: int = 0
        output_points.from_numpy(target_points)
    else:
        match_res = re.match(pattern=r".*-(\d+).obj", string=initial_guess_filepath)
        assert match_res
        iter_start: int = int(match_res.group(1))
        np_output_points, output_faces = read_obj(initial_guess_filepath)
        output_points.from_numpy(np_output_points)
    adam = Adam(
        loss_fn=compute_loss,
        loss=loss,
        x=(output_points,),
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        eta=eta,
    )
    stem = Path(output_filepath).stem

    def callback(i: int) -> bool:
        if i & (i - 1) == 0:
            write_obj(
                points=output_points.to_numpy(),
                faces=faces.to_numpy(),
                filepath=Path(output_filepath).with_stem(f"{stem}-{i}"),
            )
        return False

    try:
        adam.run(iters=iters, iter_start=iter_start, callback=callback)
    except KeyboardInterrupt:
        pass
    finally:
        write_obj(
            points=output_points.to_numpy(),
            faces=faces.to_numpy(),
            filepath=output_filepath,
        )


if __name__ == "__main__":
    typer.run(main)
