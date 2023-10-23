from pathlib import Path
from typing import Annotated

import trimesh
from trimesh import Trimesh
from typer import Argument

from taichi_extras.common.typing import cast


def main(filepath: Annotated[Path, Argument(exists=True, dir_okay=False)]) -> None:
    mesh: Trimesh = cast(Trimesh, trimesh.load(filepath))
    print(mesh)
    print("watertight:", mesh.is_watertight)


if __name__ == "__main__":
    from taichi_extras.common.typer import run

    run(main)
