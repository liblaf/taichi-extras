from pathlib import Path
from typing import Annotated, cast

import pyvista as pv
import typer
from pyvista import PolyData


def main(
    output: Annotated[
        Path, typer.Argument(dir_okay=False, writable=True, readable=False)
    ]
) -> None:
    mesh: PolyData = cast(
        PolyData, pv.Box(bounds=(-1.0, 1.0, -0.5, 0.5, -0.5, 0.5), level=4)
    )
    mesh.save(output)


if __name__ == "__main__":
    typer.run(main)
