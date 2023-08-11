import taichi as ti


@ti.func
def positive_singular_value_decomposition_func(
    mat3: ti.math.mat3,
) -> tuple[ti.math.mat3, ti.math.mat3, ti.math.mat3]:
    """
    mat3 = U @ Sigma @ V.transpose()
    """
    U, Sigma, V = ti.svd(mat3)
    if ti.math.determinant(U) < 0.0:  # type: ignore
        U[:, 2] *= -1
        Sigma[2, 2] *= -1
    if ti.math.determinant(V) < 0.0:  # type: ignore
        V[:, 2] *= -1
        Sigma[2, 2] *= -1
    return U, Sigma, V
