import typing

import taichi as ti


def reshape(
    src: ti.ScalarField | ti.MatrixField,
    new_shape: int | tuple[int, ...] = -1,
    output: typing.Optional[ti.ScalarField | ti.MatrixField] = None,
) -> ti.ScalarField | ti.MatrixField:
    if output is None:
        match src:
            case ti.ScalarField():
                res = ti.field(dtype=src.dtype, shape=new_shape)
            case ti.MatrixField():
                res = ti.Matrix.field(
                    n=src.n, m=src.m, dtype=src.dtype, shape=new_shape, ndim=src.ndim
                )
    else:
        res = output
    res.from_numpy(src.to_numpy().reshape(new_shape))
    return res
