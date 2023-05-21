import taichi as ti


@ti.func
def positive_singular_value_decomposition(
    mat3: ti.math.mat3,
) -> tuple[ti.math.mat3, ti.math.mat3, ti.math.mat3]:
    """
    mat3 = U @ Sigma @ V.transpose()
    """
    U, Sigma, V = ti.svd(mat3)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        Sigma[2, 2] *= -1
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        Sigma[2, 2] *= -1
    return U, Sigma, V
