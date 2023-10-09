from collections.abc import Callable
from typing import Any

import taichi as ti
from taichi import MatrixField, ScalarField

from taichi_extras.io import ele
from taichi_extras.io.tetgen import TetMesh
from taichi_extras.lang.mesh import mesh_element_field

from . import lhs
from .config import Component, Config, VertAttrRange


def get_cell_attr(config: Config, key: str) -> Callable[[int], Any]:
    def func(region_number: int) -> Any:
        component: Component = config.find_by_region_number(region_number)
        return getattr(component.cell_attrs, key)

    return func


def init_cell_attrs(mesh: TetMesh, config: Config) -> None:
    mesh_element_field.place(
        field=mesh.instance.cells,
        members={
            "mass_density": float,
            "stiffness": float,
            "strain_limit": ti.math.vec2,
        },
    )
    mass_density: ScalarField = mesh.instance.cells.get_member_field("mass_density")
    mass_density.from_numpy(
        ele.region_to_attrs(
            regions=mesh.cell_attrs,
            func=get_cell_attr(config=config, key="mass_density"),
        )
    )
    stiffness: ScalarField = mesh.instance.cells.get_member_field("stiffness")
    stiffness.from_numpy(
        ele.region_to_attrs(
            regions=mesh.cell_attrs,
            func=get_cell_attr(config=config, key="stiffness"),
        )
    )
    strain_limit: MatrixField = mesh.instance.cells.get_member_field("strain_limit")
    strain_limit.from_numpy(
        ele.region_to_attrs(
            regions=mesh.cell_attrs,
            func=get_cell_attr(config=config, key="strain_limit"),
        )
    )


def init_vert_attrs(mesh: TetMesh) -> None:
    mesh_element_field.place(
        field=mesh.instance.verts,
        members={
            "color_preset": ti.math.vec3,
            "position_fixed": ti.math.vec3,
            "stiffness_fixed": float,
        },
    )
    color_preset: MatrixField = mesh.instance.verts.get_member_field("color_preset")
    color_preset.from_numpy(mesh.vert_attrs[:, VertAttrRange.COLOR] / 255.0)
    position_fixed: MatrixField = mesh.instance.verts.get_member_field("position_fixed")
    position_fixed.from_numpy(mesh.vert_attrs[:, VertAttrRange.POSITION_FIXED])
    stiffness_fixed: ScalarField = mesh.instance.verts.get_member_field(
        "stiffness_fixed"
    )
    stiffness_fixed.from_numpy(mesh.vert_attrs[:, VertAttrRange.STIFFNESS_FIXED])


def init(mesh: TetMesh, config: Config) -> None:
    mesh_element_field.place(
        field=mesh.instance.verts,
        members={
            "position": ti.math.vec3,
            "position_init": ti.math.vec3,
        },
    )
    position: MatrixField = mesh.instance.verts.get_member_field("position")
    position.from_numpy(mesh.instance.get_position_as_numpy())
    position_init: MatrixField = mesh.instance.verts.get_member_field("position_init")
    position_init.copy_from(position)
    init_cell_attrs(mesh=mesh, config=config)
    init_vert_attrs(mesh=mesh)
    lhs.compute_A(mesh=mesh.instance, constants=config.constants)
