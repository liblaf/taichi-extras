from pathlib import Path

import numpy as np
import taichi as ti
import typer

from taichi_extras.mesh.obj import read_obj, write_obj
from taichi_extras.spatial import transform as tie_transform

ti.init(kernel_profiler=True)


num_points: int
num_faces: int
input_points: ti.MatrixField
output_points: ti.MatrixField
faces: ti.MatrixField


def init_fields(
    np_points: np.ndarray,
    np_faces: np.ndarray,
) -> None:
    global num_points, num_faces, input_points, output_points, faces
    num_points = np_points.shape[0]
    num_faces = np_faces.shape[0]
    input_points = ti.Vector.field(n=3, dtype=float, shape=num_points)
    output_points = ti.Vector.field(n=3, dtype=float, shape=num_points)
    faces = ti.Vector.field(n=3, dtype=int, shape=num_faces)
    input_points.from_numpy(np_points)
    faces.from_numpy(np_faces)


def main(
    input_filepath: str = typer.Option(Path.cwd() / "data" / "u0.obj", "-i", "--input"),
    output_filepath: str = typer.Option(
        Path.cwd() / "data" / "u0-aligned.obj", "-o", "--output"
    ),
    transform_filepath: str = typer.Option(
        Path.cwd() / "data" / "su-transform.txt", "-t", "--transform"
    ),
    inverse: bool = typer.Option(False),
):
    np_input_points, np_input_faces = read_obj(input_filepath)
    init_fields(
        np_points=np_input_points,
        np_faces=np_input_faces,
    )
    params = np.loadtxt(transform_filepath)
    transform, displacement = tie_transform.to_matrix(params=params)
    if inverse:
        tie_transform.inverse_transform_mesh(
            transform=transform,
            displacement=displacement,
            input_points=input_points,
            output_points=output_points,
        )
    else:
        tie_transform.transform_mesh(
            transform=transform,
            displacement=displacement,
            input_points=input_points,
            output_points=output_points,
        )
    write_obj(
        points=output_points.to_numpy(),
        faces=faces.to_numpy(),
        filepath=output_filepath,
    )

    ti.profiler.print_scoped_profiler_info()
    ti.profiler.print_kernel_profiler_info()


if __name__ == "__main__":
    typer.run(main)
