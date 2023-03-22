import taichi as ti


def like(
    x: ti.ScalarField | ti.MatrixField,
    needs_grad: bool = False,
    needs_dual: bool = False,
    *args,
    **kwargs,
) -> ti.ScalarField | ti.MatrixField:
    match x:
        case ti.ScalarField():
            return ti.field(
                dtype=x.dtype,
                shape=x.shape,
                needs_grad=needs_grad,
                needs_dual=needs_dual,
                *args,
                **kwargs,
            )
        case ti.MatrixField():
            return ti.Matrix.field(
                n=x.n,
                m=x.m,
                dtype=x.dtype,
                shape=x.shape,
                needs_grad=needs_grad,
                needs_dual=needs_dual,
                ndim=x.ndim,
                *args,
                **kwargs,
            )
