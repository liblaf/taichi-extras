from pathlib import Path
from typing import Annotated, cast

import pyvista as pv
import typer
from pymeshfix import MeshFix
from pyvista import PolyData

from taichi_extras.typer.run import run as typer_run


def main(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[
        Path, typer.Argument(dir_okay=False, writable=True, readable=False)
    ],
    *,
    binary: Annotated[bool, typer.Option()] = True,
    join_comp: Annotated[
        bool, typer.Option(help="Attempts to join nearby open components.")
    ] = False,
    remove_smallest_components: Annotated[
        bool,
        typer.Option(
            help="Remove all but the largest isolated component from the mesh before beginning the repair process."
        ),
    ] = True,
    verbose: Annotated[
        bool, typer.Option(help="Enables or disables debug printing.")
    ] = False,
) -> None:
    mesh: PolyData = cast(PolyData, pv.read(input))
    mesh_fix: MeshFix = MeshFix(mesh)
    mesh_fix.repair(
        verbose=verbose,
        joincomp=join_comp,
        remove_smallest_components=remove_smallest_components,
    )
    mesh = mesh_fix.mesh
    mesh.clean(inplace=True)
    mesh.triangulate(inplace=True)
    mesh.save(output, binary=binary)


if __name__ == "__main__":
    typer_run(main)
