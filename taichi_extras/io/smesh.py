"""
https://www.wias-berlin.de/software/tetgen/fformats.smesh.html
"""

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np


def write(
    filepath: Path,
    points: np.ndarray,
    facets: np.ndarray,
    holes: Optional[np.ndarray] = None,
    regions: Optional[np.ndarray] = None,
    *,
    point_attrs: Optional[np.ndarray] = None,
    point_boundary_markers: Optional[np.ndarray] = None,
    facet_boundary_markers: Optional[np.ndarray] = None,
    region_numbers: Optional[np.ndarray] = None,
    region_attrs: Optional[np.ndarray] = None,
    format_float: Callable[[float], str] = lambda x: format(x, ".18e"),
    format_int: Callable[[int], str] = lambda x: format(x, "d"),
) -> None:
    with filepath.open(mode="w") as file:
        print("# Part 1 - node list", file=file)
        # <# of points> <dimension (must be 3)> <# of attrs> <# of boundary markers (0 or 1)>
        print(
            points.shape[0],
            3,
            0 if point_attrs is None else point_attrs.shape[1],
            0 if point_boundary_markers is None else 1,
            file=file,
        )
        for i in range(points.shape[0]):
            # <point #> <x> <y> <z> [attrs] [boundary marker]
            print(i, *map(format_float, points[i]), end="", file=file)
            if point_attrs is not None:
                print("", *map(format_float, point_attrs[i]), end="", file=file)
            if point_boundary_markers is not None:
                print("", format_int(point_boundary_markers[i]), end="", file=file)
            print(file=file)

        print(file=file)
        print("# Part 2 - facet list", file=file)
        # <# of facets> <boundary markers (0 or 1)>
        print(facets.shape[0], 0 if facet_boundary_markers is None else 1, file=file)
        for i in range(facets.shape[0]):
            # <# of corners> <corner 1> <corner 2> ... <corner #> [boundary marker]
            print(facets.shape[1], *map(format_int, facets[i]), end="", file=file)
            if facet_boundary_markers is not None:
                print("", format_int(facet_boundary_markers[i]), end="", file=file)
            print(file=file)

        print(file=file)
        print("# Part 3 - hole list", file=file)
        if holes is not None:
            # <# of holes>
            print(holes.shape[0], file=file)
            for i in range(holes.shape[0]):
                # <hole #> <x> <y> <z>
                print(i, *map(format_float, holes[i]), file=file)
        else:
            # <# of holes>
            print(0, file=file)

        if regions is not None:
            print(file=file)
            print("# Part 4 - region attrs list", file=file)
            # <# of region>
            print(regions.shape[0], file=file)
            for i in range(regions.shape[0]):
                # <region #> <x> <y> <z> <region number> <region attribute>
                print(i, *map(format_float, regions[i]), end="", file=file)
                if region_numbers is not None:
                    print("", format_int(region_numbers[i]), end="", file=file)
                if region_attrs is not None:
                    print("", format_float(region_attrs[i]), end="", file=file)
                print(file=file)
