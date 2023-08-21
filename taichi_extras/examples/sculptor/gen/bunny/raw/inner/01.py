from pathlib import Path
from typing import Annotated, cast

import pyvista as pv
import typer
from pyvista import PolyData


def main(
    output: Annotated[
        Path,
        typer.Argument(
            exists=False, file_okay=True, dir_okay=False, writable=True, readable=False
        ),
    ]
) -> None:
    mesh: PolyData = cast(PolyData, pv.Sphere(radius=0.02, center=(-0.04, -0.02, 0.01)))
    mesh.save(output)


if __name__ == "__main__":
    typer.run(main)
