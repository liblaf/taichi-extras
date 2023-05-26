import numpy as np
import pytest
import taichi as ti

from ..math import positive_singular_value_decomposition_func
from . import EPS

ti.init()


@ti.kernel
def positive_singular_value_decomposition_kernel(
    mat3: ti.math.mat3, results: ti.template()
):
    results[0], results[1], results[2] = positive_singular_value_decomposition_func(
        mat3
    )


def positive_singular_value_decomposition(
    mat3: ti.Matrix,
) -> tuple[ti.Matrix, ti.Matrix, ti.Matrix]:
    results: ti.MatrixField = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(3,))
    positive_singular_value_decomposition_kernel(mat3=mat3, results=results)
    return results[0], results[1], results[2]


@pytest.mark.parametrize("num_exec", range(8))
def test_positive_singular_value_decomposition(num_exec) -> None:
    EPS: float = 2e-6
    mat3: ti.Matrix = ti.Matrix(np.random.rand(3, 3))
    U, Sigma, V = positive_singular_value_decomposition(mat3=mat3)
    for i in range(3):
        for j in range(3):
            if i == j:
                assert Sigma[i, i] >= 0
            else:
                np.testing.assert_allclose(Sigma[i, j], 0.0, rtol=EPS, atol=EPS)
    np.testing.assert_allclose(
        (U @ Sigma @ V.transpose() - mat3).norm(), 0.0, rtol=EPS, atol=EPS
    )
