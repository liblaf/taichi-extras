from pathlib import Path
from typing import Annotated

import numpy as np
import trimesh
from numpy.typing import NDArray
from trimesh import Trimesh
from typer import Argument, Option

from taichi_extras.common.typing import cast


def main(
    source_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_filepath: Annotated[Path, Argument(exists=True, dir_okay=False)],
) -> None:
    source: Trimesh = cast(Trimesh, trimesh.load(source_filepath))
    target: Trimesh = cast(Trimesh, trimesh.load(target_filepath))
    source_landmarks: NDArray = np.loadtxt(source_filepath.with_suffix(".landmark.txt"))
    target_landmarks: NDArray = np.loadtxt(target_filepath.with_suffix(".landmark.txt"))


if __name__ == "__main__":
    from taichi_extras.common.typer import run

    run(main)
