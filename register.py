import typing
from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
import trimesh
import typer
from numpy.typing import NDArray
from pyvista.plotting.plotter import Plotter
from trimesh import registration
from trimesh.base import Trimesh
from trimesh.points import PointCloud


def main(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    target: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    landmarks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    trimesh.util.attach_to_log()
    source_mesh: Trimesh = typing.cast(Trimesh, trimesh.load(source))
    target_mesh: PointCloud = typing.cast(PointCloud, trimesh.load(target))
    landmarks_numpy: NDArray = np.loadtxt(landmarks, dtype=int)
    matrix, transformed, cost = registration.procrustes(
        source_mesh.vertices[landmarks_numpy[0, :]],
        target_mesh.vertices[landmarks_numpy[1, :]],
    )
    source_mesh.apply_transform(matrix)
    source_mesh.export("aligned.ply")
    result: NDArray = typing.cast(
        NDArray,
        registration.nricp_amberg(
            source_mesh=source_mesh,
            target_geometry=target_mesh,
            source_landmarks=landmarks_numpy[0, :],
            target_positions=target_mesh.vertices[landmarks_numpy[1, :]],
            steps=[
                # [0.01, 10, 0.5, 10],
                # [0.02, 5, 0.5, 10],
                # [0.03, 2.5, 0.5, 10],
                # [0.01, 0, 0.0, 10],
                [0.02, 3.0, 0.5, 10],
                [0.007, 3.0, 0.5, 10],
                [0.002, 3.0, 0.5, 10],
            ],
            use_faces=False,
        ),
    )
    source_mesh.vertices = result
    source_mesh.export("registered.ply")


if __name__ == "__main__":
    typer.run(main)
