import taichi as ti


@ti.func
def positive_singular_value_decomposition_func(
    mat3: ti.math.mat3,
) -> tuple[ti.math.mat3, ti.math.mat3, ti.math.mat3]:
    """
    mat3 = U @ Sigma @ V.transpose()
    """
    U, Sigma, V = ti.svd(mat3)
    for i in ti.static(range(3)):
        if Sigma[i, i] < 0:
            Sigma[i, i] *= -1
            V[:, i] *= -1
    return U, Sigma, V


def a_add_b_mul_c(
    result: ti.MatrixField, a: ti.MatrixField, b: float, c: ti.MatrixField
) -> None:
    a_add_b_mul_c_kernel(result=result, a=a, b=b, c=c)


@ti.kernel
def a_add_b_mul_c_kernel(
    result: ti.template(), a: ti.template(), b: ti.f32, c: ti.template()
):
    for i in result:
        result[i] = a[i] + b * c[i]


def dot(a: ti.MatrixField, b: ti.MatrixField) -> float:
    return dot_kernel(a=a, b=b)


@ti.kernel
def dot_kernel(a: ti.template(), b: ti.template()) -> ti.f32:
    result = 0.0
    for i in a:
        result += a[i].dot(b[i])
    return result
