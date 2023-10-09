from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray
from pyvista.core.pointset import PolyData
from pyvista.plotting.plotter import Plotter

from taichi_extras.io import pyvista as io_pv


def main(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    target: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    landmarks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    source_mesh: PolyData = io_pv.read_poly_data(source)
    target_mesh: PolyData = io_pv.read_poly_data(target)
    landmarks_numpy: NDArray = np.loadtxt(landmarks, dtype=int)
    plotter: Plotter = Plotter(window_size=[768, 1024])
    plotter.add_mesh(source_mesh, opacity=0.5)
    plotter.add_points(target_mesh.points, point_size=1, opacity=0.5)
    plotter.add_point_labels(
        points=source_mesh.points[landmarks_numpy[0, :]],
        # labels=landmarks_numpy[0, :],
        labels=[""] * landmarks_numpy.shape[1],
        point_color="red",
        point_size=8,
        render_points_as_spheres=True,
        always_visible=True,
    )
    plotter.add_point_labels(
        points=target_mesh.points[landmarks_numpy[1, :]],
        # labels=landmarks_numpy[1, :],
        labels=[""] * landmarks_numpy.shape[1],
        point_color="green",
        point_size=8,
        render_points_as_spheres=True,
        always_visible=True,
    )
    plotter.view_xy()  # type: ignore
    plotter.save_graphic("aligned.svg")
    plotter.show()


if __name__ == "__main__":
    typer.run(main)
