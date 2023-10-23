from pathlib import Path
from typing import Annotated

import trimesh
from trimesh import Trimesh
from typer import Argument, Option

from taichi_extras.common.typing import cast


def main(
    input_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    output_path: Annotated[Path, Argument(dir_okay=False, writable=True)],
    *,
    face_count: Annotated[int, Option()],
) -> None:
    mesh: Trimesh = cast(Trimesh, trimesh.load(input_path))
    mesh = mesh.simplify_quadric_decimation(face_count=face_count)
    mesh.export(output_path)


if __name__ == "__main__":
    from taichi_extras.common.typer import run

    run(main)
