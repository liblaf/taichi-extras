"""
https://www.wias-berlin.de/software/tetgen/fformats.ele.html
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .utils import minify


def read(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    text: str = filepath.read_text()
    lines: list[str] = list(minify(text.splitlines()))
    # <# of tetrahedra> <nodes per tetrahedron> <# of attrs>
    num_tets, nodes_per_tet, num_attrs = map(int, lines[0].split())
    tets: np.ndarray = np.zeros(shape=(num_tets, nodes_per_tet), dtype=int)
    attrs: np.ndarray = np.zeros(shape=(num_tets, num_attrs), dtype=int)
    for line in lines[1 : num_tets + 1]:
        # <tetrahedron #> <node> <node> <node> <node> ... [attrs]
        words: list[str] = line.split()
        index: int = int(words[0])
        tets[index] = list(map(int, words[1 : nodes_per_tet + 1]))
        attrs[index] = list(map(int, words[nodes_per_tet + 1 :]))
    return tets, attrs


def region_to_attrs(regions: np.ndarray, func: Callable[[int], Any]) -> np.ndarray:
    regions = np.reshape(regions, newshape=(-1, 1))
    return np.apply_along_axis(lambda x: func(x[0]), axis=1, arr=regions)
