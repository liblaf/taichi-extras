import logging
import typing
from pathlib import Path
from typing import Any

import pyvista as pv
import trimesh
from pymeshfix import MeshFix
from pyvista.core.composite import MultiBlock
from pyvista.core.pointset import PolyData
from rich.logging import RichHandler
from trimesh import boolean
from trimesh.base import Trimesh

import taichi_extras.logging
from taichi_extras.io import pyvista as io_pv


def aaaa(typ, val):
    assert isinstance(val, typ)
    return val


taichi_extras.logging.init(level=logging.DEBUG)
logger = logging.getLogger("taichi_extras")
logger.info("Hello, world!")
mesh: PolyData = io_pv.read_poly_data(
    Path.cwd()
    / "taichi_extras"
    / "examples"
    / "sculptor"
    / "data"
    / "chin"
    / "raw"
    / "skull.ply"
)
mesh.triangulate(inplace=True)
bodies: MultiBlock = mesh.split_bodies(progress_bar=True).as_polydata_blocks()
mandible = typing.cast(PolyData, bodies[0])
maxilla = aaaa(PolyData, bodies[1])
mandible_fix = MeshFix(mandible)
mandible_fix.repair(verbose=False)
mandible = mandible_fix.mesh
maxilla_fix = MeshFix(maxilla)
maxilla_fix.repair(verbose=False)
maxilla = maxilla_fix.mesh
mandible = Trimesh(vertices=mandible.points, faces=mandible.faces.reshape(-1, 4)[:, 1:])
maxilla = Trimesh(vertices=maxilla.points, faces=maxilla.faces.reshape(-1, 4)[:, 1:])
result: Trimesh = boolean.union([mandible, maxilla])
result.export("skull.ply")
result: PolyData = pv.wrap(result)
print(type(result))
# result_fix: MeshFix = MeshFix(pv.wrap(result))
# result_fix.repair()
# result = result_fix.mesh
# result.save("skull.ply")
# mesh_fix: MeshFix = MeshFix(mesh)
# mesh_fix.repair(joincomp=True, remove_smallest_components=True)
# mesh = mesh_fix.mesh
# mesh.save("skull.ply")

# print(result)
