from pathlib import Path
from typing import Annotated

import trimesh
import typer
from numpy.typing import NDArray
from trimesh import Trimesh
from typer import Argument

from taichi_extras.common.typing import cast


def main(
    input_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    output_path: Annotated[Path, Argument(dir_okay=False, writable=True)],
) -> None:
    mesh: Trimesh = cast(Trimesh, trimesh.load(input_path))
    bodies: NDArray = cast(NDArray, mesh.split())
    mesh = bodies[0]
    mesh.export(output_path)


if __name__ == "__main__":
    typer.run(main)
