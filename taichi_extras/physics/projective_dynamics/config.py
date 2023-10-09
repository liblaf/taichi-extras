import functools
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel


def deep_update(old: dict, new: dict) -> dict:
    result: dict = old.copy()
    for k, v in new.items():
        if k in old and isinstance(old[k], dict) and isinstance(v, dict):
            result[k] = deep_update(old[k], v)
        else:
            result[k] = v
    return result


class Camera(BaseModel):
    lookat: tuple[float, float, float]
    position: tuple[float, float, float]


class CellAttrs(BaseModel):
    hole: bool
    mass_density: float
    region_number: int
    stiffness: float
    strain_limit: tuple[float, float]


class PointsFixed(StrEnum):
    ALL = "all"


class VertAttrRange:
    COLOR = range(0, 3)
    POSITION_FIXED = range(3, 6)
    STIFFNESS_FIXED = 6
    LEN = 7


class VertAttrs(BaseModel):
    color: tuple[int, int, int]
    point_boundary_marker: int
    points_fixed: PointsFixed | Sequence[int]
    stiffness_fixed: float


class Component(BaseModel):
    filename: str = ""
    cell_attrs: CellAttrs
    vert_attrs: VertAttrs


class Constants(BaseModel):
    gravity: tuple[float, float, float]
    time_step: float
    tolerance: float


class Config(BaseModel):
    camera: Camera
    components: Sequence[Component]
    constants: Constants
    default: Component

    @classmethod
    def load_yaml(cls, filepath: Path) -> Self:
        config_dict: dict[str, Any] = yaml.safe_load(filepath.read_text())
        config_dict["components"] = [
            deep_update(config_dict["default"], component)
            for component in config_dict["components"]
        ]
        return Config(**config_dict)

    @functools.lru_cache()
    def find_by_region_number(self, region_number: int) -> Component:
        try:
            return next(
                component
                for component in self.components
                if component.cell_attrs.region_number == region_number
            )
        except StopIteration:
            return self.default

    def __hash__(self) -> int:
        return id(self)
