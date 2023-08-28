from io import BytesIO
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import requests
import typer
from pyvista import PolyData
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
    face.save(output / "outer.ply")
    for i, block in enumerate(skull.split_bodies().as_polydata_blocks()):
        mesh: PolyData = cast(PolyData, block)
        mesh.save(output / "inner" / f"{i:02d}.ply")


if __name__ == "__main__":
    typer.run(main)
