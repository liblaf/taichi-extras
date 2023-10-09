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
        PolyData, pv.Box(bounds=(-0.9, 0.9, 0.3, 0.4, -0.4, 0.4), level=4)
    )
    mesh.save(output)


if __name__ == "__main__":
    typer.run(main)
