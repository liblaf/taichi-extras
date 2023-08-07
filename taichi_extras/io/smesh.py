"""
https://www.wias-berlin.de/software/tetgen/fformats.smesh.html
"""

from pathlib import Path
from typing import Optional

import numpy as np


def write(
    filepath: Path,
    points: np.ndarray,
    facets: np.ndarray,
    holes: Optional[np.ndarray] = None,
    *,
    fmt: str = ".18e",
) -> None:
    if holes is None:
        holes = np.empty(shape=(0, 3))

    assert points.ndim == 2 and points.shape[1] == 3
    assert facets.ndim == 2 and points.shape[1] == 3
    assert holes.ndim == 2 and holes.shape[1] == 3

    with filepath.open(mode="w") as file:
        print("# Part 1 - node list", file=file)
        print(points.shape[0], 3, 0, 0, file=file)
        for i in range(points.shape[0]):
            print(i, *map(lambda x: format(x, fmt), points[i]), file=file)

        print(file=file)
        print("# Part 2 - facet list", file=file)
        print(facets.shape[0], 0, file=file)
        for i in range(facets.shape[0]):
            print(3, *facets[i], file=file)

        print(file=file)
        print("# Part 3 - hole list", file=file)
        print(f"{holes.shape[0]}", file=file)
        for i in range(holes.shape[0]):
            print(i, *map(lambda x: format(x, fmt), holes[i]), file=file)

        print(file=file)
        print("# Part 4 - region attributes list", file=file)
        print(0, file=file)
