from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from pyvista import PolyData
from pyvista.plotting.plotter import Plotter
from typer import Argument

from taichi_extras.common.typing import cast


def main(
    filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
) -> None:
    plotter: Plotter = Plotter()
    source: PolyData = cast(PolyData, pv.read(filepath))
    source_landmarks: NDArray = np.loadtxt(fname=filepath.with_suffix(".landmark.txt"))
    plotter.add_mesh(source, opacity=0.2)
    plotter.add_point_labels(
        points=source_landmarks,
        labels=range(source_landmarks.shape[0]),
        point_color="green",
        point_size=source.length / 20,
        render_points_as_spheres=True,
        always_visible=True,
    )
    plotter.show()


if __name__ == "__main__":
    import taichi_extras.common.typer

    taichi_extras.common.typer.run(main)
