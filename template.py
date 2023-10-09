from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pyvista as pv
import requests
import typer
from numpy.typing import NDArray
from pymeshfix import MeshFix
from pyvista import MultiBlock, PolyData
from requests import Response
from trimesh.base import Trimesh

from taichi_extras.pyvista import poly_data

URL_PREFIX: str = "https://github.com/liblaf/sculptor/raw/main/data"


def download_numpy(url: str) -> np.ndarray:
    response: Response = requests.get(url=url, stream=True)
    response.raise_for_status()
    data: np.ndarray = np.load(BytesIO(response.content))
    return data


def main(
    output: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=False, dir_okay=True, writable=True, readable=False
        ),
    ]
) -> None:
    face: Trimesh = Trimesh(
        vertices=download_numpy(f"{URL_PREFIX}/template_face.npy"),
        faces=download_numpy(f"{URL_PREFIX}/face_face.npy"),
    )
    skull: Trimesh = Trimesh(
        vertices=download_numpy(f"{URL_PREFIX}/template_skull.npy"),
        faces=download_numpy(f"{URL_PREFIX}/skull_face.npy"),
    )
    face.export(output / "face-raw.ply")
    face = cast(
        Trimesh,
        face.slice_plane(plane_origin=[0.0, 0.0, -50.0], plane_normal=[0.0, 0.0, 1.0]),
    )
    mesh_fix: MeshFix = MeshFix(pv.wrap(face))
    mesh_fix.repair()
    mesh_fix.save(output / "face.ply")
    # face.export(output / "face.ply")
    blocks: Sequence[Trimesh] = cast(Sequence[Trimesh], skull.split())
    mandible: Trimesh = cast(Trimesh, blocks[0])
    maxilla: Trimesh = cast(Trimesh, blocks[1])
    skull = cast(Trimesh, mandible.union(maxilla))
    mandible.export(output / "mandible.ply")
    maxilla.export(output / "maxilla.ply")
    skull.export(output / "skull.ply")
    # maxilla = cast(PolyData, maxilla.boolean_difference(mandible))
    # maxilla.clip_closed_surface(normal="y", origin=[0.0, -20.0, 0.0], inplace=True)


if __name__ == "__main__":
    typer.run(main)
