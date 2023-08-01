from pathlib import Path
from typing import Optional

import numpy as np


def write(
    filepath: Path,
    points: np.ndarray,
    facets: np.ndarray,
    holes: Optional[np.ndarray] = None,
    attributes: Optional[np.ndarray] = None,
) -> None:
    """
    https://www.wias-berlin.de/software/tetgen/fformats.smesh.html
    """
    if holes is None:
        holes = np.empty(shape=(0, 3))
    if attributes is None:
        attributes = np.empty(shape=(points.shape[0], 0))

    assert points.ndim == 2 and points.shape[1] == 3
    assert facets.ndim == 2 and points.shape[1] == 3
    assert holes.ndim == 2 and holes.shape[1] == 3
    assert attributes.ndim == 2 and attributes.shape[0] == points.shape[0]

    with filepath.open(mode="w") as file:
        print("# Part 1 - node list", file=file)
        print(f"{points.shape[0]} 3 {attributes.shape[1]} 0", file=file)
        for i in range(points.shape[0]):
            print(
                f"{i} {points[i, 0]} {points[i, 1]} {points[i, 2]} {' '.join(attributes[i])}",
                file=file,
            )

        print("# Part 2 - facet list", file=file)
        print(f"{facets.shape[0]} 0", file=file)
        for i in range(facets.shape[0]):
            print(f"3 {facets[i, 0]} {facets[i, 1]} {facets[i, 2]}", file=file)

        print("# Part 3 - hole list", file=file)
        print(f"{holes.shape[0]}", file=file)
        for i in range(holes.shape[0]):
            print(f"{i} {holes[i, 0]} {holes[i, 1]} {holes[i, 2]}", file=file)

        print("# Part 4 - region attributes list", file=file)
        print("0", file=file)
