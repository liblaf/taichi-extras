from pathlib import Path
from typing import Annotated, cast

import typer
from pyvista import PolyData, examples


def main(
    output: Annotated[
        Path,
        typer.Argument(
            exists=False, file_okay=True, dir_okay=False, writable=True, readable=False
        ),
    ]
) -> None:
    mesh: PolyData = cast(PolyData, examples.download_bunny(load=True))
    mesh.points -= mesh.center
    mesh.flip_normals()
    mesh.save(output)


if __name__ == "__main__":
    typer.run(main)
