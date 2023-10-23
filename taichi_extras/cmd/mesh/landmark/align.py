from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pyvista as pv
import trimesh.registration
from numpy.typing import NDArray
from pyvista import PolyData
from pyvista.plotting.plotter import Plotter
from typer import Argument, Option

from taichi_extras.common.typing import cast


def main(
    source_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    *,
    output_path: Annotated[
        Optional[Path], Option("-o", "--output", dir_okay=False, writable=True)
    ] = None,
    view: Annotated[bool, Option()] = False,
) -> None:
    source: PolyData = cast(PolyData, pv.read(source_path))
    target: PolyData = cast(PolyData, pv.read(target_path))
    source_landmarks: NDArray = np.loadtxt(
        fname=source_path.with_suffix(".landmark.txt")
    )
    target_landmarks: NDArray = np.loadtxt(
        fname=target_path.with_suffix(".landmark.txt")
    )
    matrix, source_landmarks, cost = trimesh.registration.procrustes(
        a=source_landmarks, b=target_landmarks
    )
    source.transform(matrix)
    if output_path:
        source.save(output_path)
        np.savetxt(output_path.with_suffix(".landmark.txt"), source_landmarks)
    if view:
        plotter: Plotter = Plotter()
        plotter.add_mesh(source, color="red", opacity=0.2)
        plotter.add_point_labels(
            points=source_landmarks,
            labels=range(source_landmarks.shape[0]),
            point_color="red",
            point_size=source.length / 20,
            render_points_as_spheres=True,
            always_visible=True,
        )
        plotter.add_mesh(target, color="green", opacity=0.2)
        plotter.add_point_labels(
            points=target_landmarks,
            labels=range(target_landmarks.shape[0]),
            point_color="green",
            point_size=target.length / 20,
            render_points_as_spheres=True,
            always_visible=True,
        )
        plotter.show()


if __name__ == "__main__":
    import taichi_extras.common.typer

    taichi_extras.common.typer.run(main)
