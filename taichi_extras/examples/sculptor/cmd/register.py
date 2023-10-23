from pathlib import Path
from typing import Annotated

import numpy as np
import trimesh.proximity
import trimesh.registration
from numpy.typing import NDArray
from trimesh import Trimesh
from typer import Argument, Option

from taichi_extras.common.typing import cast


def main(
    source_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    target_path: Annotated[Path, Argument(exists=True, dir_okay=False)],
    *,
    output_path: Annotated[
        Path, Option("-o", "--output", dir_okay=False, writable=True)
    ]
) -> None:
    source: Trimesh = cast(Trimesh, trimesh.load(source_path))
    target: Trimesh = cast(Trimesh, trimesh.load(target_path))
    source_landmarks: NDArray = np.loadtxt(
        fname=source_path.with_suffix(".landmark.txt")
    )
    target_landmarks: NDArray = np.loadtxt(
        fname=target_path.with_suffix(".landmark.txt")
    )
    distance, vertex_id = source.nearest.vertex(source_landmarks)
    matrix, source_landmarks, cost = trimesh.registration.procrustes(
        a=source_landmarks, b=target_landmarks
    )
    source = source.apply_transform(matrix)
    position: NDArray = cast(
        NDArray,
        trimesh.registration.nricp_amberg(
            source_mesh=source,
            target_geometry=target,
            source_landmarks=vertex_id,
            target_positions=target_landmarks,
            steps=[
                # ws, wl, wn, max_iter
                [0.02, 3, 0.5, 10],
                [0.007, 3, 0.5, 10],
                [0.002, 3, 0.5, 10],
            ],
        ),
    )
    source = Trimesh(vertices=position, faces=source.faces)
    source.export(output_path, encoding="ascii")


if __name__ == "__main__":
    import taichi_extras.common.typer

    taichi_extras.common.typer.run(main)
