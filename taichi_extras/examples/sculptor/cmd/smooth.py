from pathlib import Path
from typing import Annotated

import pyacvd
import pyvista as pv
import trimesh
import trimesh.smoothing
import typer
from pyvista import PolyData
from trimesh import Trimesh
from typer import Argument

from taichi_extras.common.typing import cast


def main(
    input_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    output_path: Annotated[Path, Argument(dir_okay=False, writable=True)],
) -> None:
    mesh: Trimesh = cast(Trimesh, trimesh.load(input_path))
    # mesh = trimesh.smoothing.filter_laplacian(mesh=mesh)
    # mesh.export(output_path)
    mesh_pv: PolyData = cast(PolyData, pv.wrap(mesh))
    # clustering = pyacvd.Clustering(mesh_pv)
    # clustering.cluster(nclus=10000)
    # mesh_pv = clustering.create_mesh()
    mesh_pv.smooth_taubin(
        n_iter=100,
        boundary_smoothing=True,
        feature_smoothing=True,
        inplace=True,
        progress_bar=True,
    )
    mesh_pv.save(output_path)


if __name__ == "__main__":
    typer.run(main)
