import numpy as np
import pytest
import taichi as ti

from ..math import a_add_b_mul_c, positive_singular_value_decomposition_func
from . import TOLERANCE

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


def test_positive_singular_value_decomposition() -> None:
    mat3: ti.Matrix = ti.Matrix(np.random.rand(3, 3))
    U, Sigma, V = positive_singular_value_decomposition(mat3=mat3)
    for i in range(3):
        for j in range(3):
            if i == j:
                assert Sigma[i, i] >= 0
            else:
                np.testing.assert_allclose(
                    Sigma[i, j], 0.0, rtol=TOLERANCE, atol=TOLERANCE
                )
    np.testing.assert_allclose(
        (U @ Sigma @ V.transpose() - mat3).norm(), 0.0, rtol=TOLERANCE, atol=TOLERANCE
    )


def test_a_add_b_mul_c() -> None:
    shape: tuple[int, ...] = (3,)
    a_numpy: np.ndarray = np.random.rand(*shape, 3)
    b: float = np.random.rand()
    c_numpy: np.ndarray = np.random.rand(*shape, 3)
    result_numpy: np.ndarray = a_numpy + b * c_numpy
    result: ti.MatrixField = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    a: ti.MatrixField = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    c: ti.MatrixField = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    a.from_numpy(a_numpy)
    c.from_numpy(c_numpy)
    a_add_b_mul_c(result=result, a=a, b=b, c=c)
    np.testing.assert_allclose(
        result.to_numpy(), result_numpy, rtol=TOLERANCE, atol=TOLERANCE
    )
