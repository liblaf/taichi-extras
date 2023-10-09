from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pyvista import PolyData

from taichi_extras.io import pyvista as io_pv
from taichi_extras.io import smesh
from taichi_extras.physics.projective_dynamics.config import (
    Component,
    Config,
    PointsFixed,
    VertAttrRange,
)
from taichi_extras.pyvista import poly_data
from taichi_extras.typer.run import run as typer_run


def make_point_attrs(component: Component, mesh: PolyData, prefix: Path) -> np.ndarray:
    colors: np.ndarray = np.full(
        shape=(mesh.n_points, len(component.vert_attrs.color)),
        fill_value=component.vert_attrs.color,
    )
    position_fixed: np.ndarray
    stiffness_fixed: np.ndarray
    if component.vert_attrs.stiffness_fixed and component.vert_attrs.points_fixed:
        mesh_fixed: PolyData = io_pv.read_poly_data(
            prefix / "after" / component.filename
        )
        position_fixed, _ = poly_data.get_points_faces(mesh_fixed)
        match component.vert_attrs.points_fixed:
            case PointsFixed.ALL:
                stiffness_fixed = np.full(
                    shape=(mesh.n_points,),
                    fill_value=component.vert_attrs.stiffness_fixed,
                )
            case points_fixed:
                stiffness_fixed = np.full(shape=(mesh.n_points,), fill_value=0.0)
                stiffness_fixed[points_fixed] = component.vert_attrs.stiffness_fixed
    else:
        position_fixed = np.full(shape=(mesh.n_points, 3), fill_value=0.0)
        stiffness_fixed = np.full(shape=(mesh.n_points,), fill_value=0.0)
    attrs: np.ndarray = np.empty(shape=(mesh.n_points, VertAttrRange.LEN))
    attrs[:, VertAttrRange.COLOR] = colors
    attrs[:, VertAttrRange.POSITION_FIXED] = position_fixed
    attrs[:, VertAttrRange.STIFFNESS_FIXED] = stiffness_fixed
    return attrs


def main(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[
        Path, typer.Argument(dir_okay=False, writable=True, readable=False)
    ],
    *,
    prefix: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    config_model: Config = Config.load_yaml(config)
    components: Sequence[Component] = config_model.components
    points: np.ndarray = np.empty(shape=(0, 3))
    point_attrs: np.ndarray = np.empty(shape=(0, VertAttrRange.LEN))
    point_boundary_markers: np.ndarray = np.empty(shape=(0,), dtype=int)
    facets: np.ndarray = np.empty(shape=(0, 3), dtype=int)
    holes: np.ndarray = np.empty(shape=(0, 3))
    regions: np.ndarray = np.empty(shape=(0, 3))
    region_numbers: np.ndarray = np.empty(shape=(0,), dtype=int)
    for component in components:
        mesh: PolyData = io_pv.read_poly_data(prefix / "before" / component.filename)
        new_points, new_facets = poly_data.get_points_faces(mesh)
        new_facets += points.shape[0]
        points = np.vstack((points, new_points))
        point_attrs = np.vstack(
            (point_attrs, make_point_attrs(component, mesh, prefix))
        )
        point_boundary_markers = np.concatenate(
            (
                point_boundary_markers,
                np.full(
                    shape=(mesh.n_points,),
                    fill_value=component.vert_attrs.point_boundary_marker,
                ),
            )
        )
        facets = np.vstack((facets, new_facets))
        if component.cell_attrs.hole:
            enclosed_point: np.ndarray = poly_data.find_enclosed_point(mesh)
            holes = np.vstack((holes, enclosed_point))
        if component.cell_attrs.region_number:
            enclosed_point: np.ndarray = poly_data.find_enclosed_point(mesh)
            regions = np.vstack((regions, enclosed_point))
            region_numbers = np.concatenate(
                (region_numbers, [component.cell_attrs.region_number])
            )
    smesh.write(
        output,
        points=points,
        facets=facets,
        holes=holes,
        regions=regions,
        point_attrs=point_attrs,
        point_boundary_markers=point_boundary_markers,
        region_numbers=region_numbers,
    )


if __name__ == "__main__":
    typer_run(main)
