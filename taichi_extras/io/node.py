"""
https://www.wias-berlin.de/software/tetgen/fformats.node.html
"""

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np

from .utils import minify


def read(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text: str = filepath.read_text()
    lines: list[str] = list(minify(text.splitlines()))
    # <# of points> <dimension (must be 3)> <# of attrs> <# of boundary markers (0 or 1)>
    num_points, dimension, num_attrs, boundary = map(int, lines[0].split())
    assert dimension == 3
    points: np.ndarray = np.zeros(shape=(num_points, 3))
    attrs: np.ndarray = np.zeros(shape=(num_points, num_attrs))
    boundary_markers: np.ndarray = np.zeros(shape=(num_points,), dtype=int)
    for line in lines[1 : num_points + 1]:
        # <point #> <x> <y> <z> [attrs] [boundary marker]
        words: list[str] = line.split()
        index: int = int(words[0])
        points[index] = list(map(float, words[1:4]))
        if num_attrs:
            attrs[index] = list(map(float, words[4 : 4 + num_attrs]))
        if boundary:
            boundary_markers[index] = int(words[4 + num_attrs])
    return points, attrs, boundary_markers


def write(
    filepath: Path,
    points: np.ndarray,
    *,
    attrs: Optional[np.ndarray] = None,
    boundary_markers: Optional[np.ndarray] = None,
    format_float: Callable[[float], str] = lambda x: format(x, ".18e"),
) -> None:
    with filepath.open(mode="w") as file:
        # <# of points> <dimension (must be 3)> <# of attrs> <# of boundary markers (0 or 1)>
        print(
            points.shape[0],
            points.shape[1],
            0 if attrs is None else attrs.shape[1],
            0 if boundary_markers is None else 1,
            file=file,
        )
        for i, p in enumerate(points):
            # <point #> <x> <y> <z> [attrs] [boundary marker]
            print(i, *map(format_float, p), end="", file=file)
            if attrs is not None:
                print("", *map(format_float, attrs[i]), end="", file=file)
            if boundary_markers is not None:
                print("", boundary_markers[i], end="", file=file)
            print(file=file)
