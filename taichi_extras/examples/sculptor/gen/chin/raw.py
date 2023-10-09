from io import BytesIO
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pyvista as pv
import requests
import typer
from pyvista import MultiBlock, PolyData
from requests import Response

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
    face: PolyData = poly_data.make_mesh(
        points=download_numpy(f"{URL_PREFIX}/template_face.npy"),
        faces=download_numpy(f"{URL_PREFIX}/face_face.npy"),
    )
    skull: PolyData = poly_data.make_mesh(
        points=download_numpy(f"{URL_PREFIX}/template_skull.npy"),
        faces=download_numpy(f"{URL_PREFIX}/skull_face.npy"),
    )

    skull.points -= face.center
    face.points -= face.center
    face.rotate_x(angle=-90.0, inplace=True)
    skull.rotate_x(angle=-90.0, inplace=True)
    face.save(output / "face.ply")
    blocks: MultiBlock = skull.split_bodies().as_polydata_blocks()
    mandible: PolyData = cast(PolyData, blocks[0])
    maxilla: PolyData = cast(PolyData, blocks[1])
    skull = cast(PolyData, mandible.boolean_union(maxilla))
    mandible.save(output / "mandible.ply")
    skull.save(output / "skull.ply")
    # maxilla = cast(PolyData, maxilla.boolean_difference(mandible))
    # maxilla.clip_closed_surface(normal="y", origin=[0.0, -20.0, 0.0], inplace=True)
    maxilla.save(output / "maxilla.ply")


if __name__ == "__main__":
    typer.run(main)
