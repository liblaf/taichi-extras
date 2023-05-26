from pathlib import Path

import numpy as np
import pytest
import taichi as ti

from taichi_extras.examples.projective_statics.main import init
from taichi_extras.examples.projective_statics.utils.const import SHEAR_MODULUS

from . import EPS

ti.init()


@pytest.fixture(scope="session")
def mesh_filepath(pytestconfig) -> Path:
    return Path(pytestconfig.getoption("mesh_filepath"))


def compute_hessian(mesh: ti.MeshInstance) -> np.ndarray:
    mesh.edges.place(members={"u": ti.i32, "v": ti.i32})
    num_verts: int = len(mesh.verts)
    hessian: ti.MatrixField = ti.Matrix.field(
        n=3, m=3, dtype=ti.f32, shape=(num_verts, num_verts)
    )
    compute_hessian_kernel(mesh=mesh, hessian=hessian)
    return hessian.to_numpy()


@ti.kernel
def compute_hessian_kernel(mesh: ti.template(), hessian: ti.template()):
    for c in mesh.cells:
        for i, j, x, y in ti.ndrange(4, 4, 3, 3):
            X_f_i = ti.Matrix.zero(dt=ti.f32, n=3, m=3)
            X_f_j = ti.Matrix.zero(dt=ti.f32, n=3, m=3)
            if i == 3:
                X_f_i[x, :] = -1
            else:
                X_f_i[x, i] = 1
            if j == 3:
                X_f_j[y, :] = -1
            else:
                X_f_j[y, j] = 1
            hessian[c.verts[i].id, c.verts[j].id][x, y] += (
                SHEAR_MODULUS
                * c.volume
                * (
                    (X_f_i @ c.undeformed_shape_inverse)
                    * (X_f_j @ c.undeformed_shape_inverse)
                ).sum()
            )
    for e in mesh.edges:
        e.u = e.verts[0].id
        e.v = e.verts[1].id


def test_hessian(mesh_filepath: Path) -> None:
    if not mesh_filepath.exists():
        pytest.skip()
    mesh, _ = init(mesh_filepath)
    expected: np.ndarray = compute_hessian(mesh=mesh)
    verts_hessian: ti.ScalarField = mesh.verts.get_member_field(key="hessian")
    edges_hessian: ti.ScalarField = mesh.edges.get_member_field(key="hessian")
    edges_u: np.ndarray = mesh.edges.get_member_field(key="u").to_numpy()
    edges_v: np.ndarray = mesh.edges.get_member_field(key="v").to_numpy()
    num_verts: int = len(mesh.verts)
    for i, j, x, y in np.ndindex(num_verts, num_verts, 3, 3):
        if x == y:
            if i == j:
                np.testing.assert_allclose(
                    verts_hessian[i], expected[i, j, x, y], rtol=EPS, atol=EPS
                )
        else:
            np.testing.assert_allclose(expected[i, j, x, y], 0.0, rtol=EPS, atol=EPS)
    for e, x, y in np.ndindex(len(edges_u), 3, 3):
        if x == y:
            i, j = edges_u[e], edges_v[e]
            np.testing.assert_allclose(
                edges_hessian[e], expected[i, j, x, y], rtol=EPS, atol=EPS
            )
        else:
            np.testing.assert_allclose(
                expected[edges_u[e], edges_v[e], x, y], 0.0, rtol=EPS, atol=EPS
            )
