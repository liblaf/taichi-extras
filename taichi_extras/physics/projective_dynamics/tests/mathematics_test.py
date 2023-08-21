import numpy as np
import taichi as ti
from taichi import Matrix, MatrixField

from ..const import TOLERANCE
from ..mathematics import positive_singular_value_decomposition_func

ti.init()


@ti.kernel
def positive_singular_value_decomposition_kernel(
    mat3: ti.math.mat3,
    results: ti.template(),  # type: ignore
):
    results[0], results[1], results[2] = positive_singular_value_decomposition_func(
        mat3
    )


def positive_singular_value_decomposition(
    mat3: Matrix,
) -> tuple[Matrix, Matrix, Matrix]:
    results: MatrixField = Matrix.field(n=3, m=3, dtype=ti.f32, shape=(3,))
    positive_singular_value_decomposition_kernel(mat3=mat3, results=results)
    return results[0], results[1], results[2]


def test_positive_singular_value_decomposition() -> None:
    mat3: Matrix = Matrix(np.random.rand(3, 3))
    U, Sigma, V = positive_singular_value_decomposition(mat3)
    assert np.linalg.det(U.to_numpy()) > 0.0  # type: ignore
    assert np.linalg.det(V.to_numpy()) > 0.0  # type: ignore
    for i, j in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(Sigma[i, j], 0.0, rtol=TOLERANCE, atol=TOLERANCE)
    result: Matrix = U @ Sigma @ V.transpose()  # type: ignore
    np.testing.assert_allclose(
        result.to_numpy(),  # type: ignore
        mat3.to_numpy(),  # type: ignore
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )
