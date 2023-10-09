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
        PolyData,
        pv.Cylinder(
            center=(-0.95, 0.45, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=0.04,
            height=0.8,
        ),
    )
    mesh.save(output)


if __name__ == "__main__":
    typer.run(main)
