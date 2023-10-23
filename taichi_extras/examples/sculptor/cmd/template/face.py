from pathlib import Path
from typing import Annotated

import trimesh
import trimesh.repair
from trimesh import Trimesh
from typer import Argument

from taichi_extras.common.typing import cast


def main(
    input_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    output_path: Annotated[Path, Argument(dir_okay=False, writable=True)],
) -> None:
    mesh: Trimesh = cast(Trimesh, trimesh.load(input_path))
    mesh = cast(
        Trimesh, mesh.slice_plane(plane_origin=[0, 0, -50], plane_normal=[0, 0, 1])
    )
    trimesh.repair.fill_holes(mesh)
    mesh.export(output_path)


if __name__ == "__main__":
    from taichi_extras.common.typer import run

    run(main)
