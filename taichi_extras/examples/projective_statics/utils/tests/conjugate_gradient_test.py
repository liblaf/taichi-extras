import numpy as np
import taichi as ti

from ..conjugate_gradient import conjugate_gradient_method
from . import TOLERANCE

ti.init()


def A_mul_x(
    result: ti.MatrixField, A: ti.MatrixField, x: ti.MatrixField
) -> ti.MatrixField:
    A_mul_x_kernel(result=result, A=A, x=x)
    return result


@ti.kernel
def A_mul_x_kernel(result: ti.template(), A: ti.template(), x: ti.template()):
    for i in result:
        result[i] = A[i] @ x[i]


def test_conjugate_gradient_method() -> None:
    shape: tuple[int, ...] = (4,)
    A: ti.MatrixField = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=shape)
    b: ti.MatrixField = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    p: ti.MatrixField = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    product: ti.MatrixField = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    r: ti.MatrixField = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    x: ti.MatrixField = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    A.from_numpy(np.random.rand(*shape, 3, 3))
    x.from_numpy(np.random.rand(*shape, 3))
    b = A_mul_x(result=b, A=A, x=x)

    def mul_x(x: ti.MatrixField) -> ti.MatrixField:
        return A_mul_x(result=product, A=A, x=x)

    x = conjugate_gradient_method(b=b, p=p, r=r, x=x, A_mul_x=mul_x, n_iter=64)
    product = A_mul_x(result=product, A=A, x=x)
    np.testing.assert_allclose(
        product.to_numpy(), b.to_numpy(), rtol=TOLERANCE, atol=TOLERANCE
    )
