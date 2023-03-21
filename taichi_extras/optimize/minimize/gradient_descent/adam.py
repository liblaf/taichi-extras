import typing

import taichi as ti


class ADAM:
    loss_fn: typing.Callable

    beta_1: float
    beta_2: float
    epsilon: float
    eta: float

    m: ti.MatrixField
    v: ti.MatrixField
    x: ti.MatrixField
    loss: ti.ScalarField

    def __init__(
        self,
        loss_fn: typing.Callable,
        loss: ti.ScalarField,
        x: ti.MatrixField,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        eta: float = 1e-3,
    ) -> None:
        self.loss_fn = loss_fn
        self.loss = loss
        self.x = x
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.eta = eta

        self.m = ti.Vector.field(n=x.n, dtype=x.dtype, shape=x.shape, needs_grad=False)
        self.v = ti.Vector.field(n=x.n, dtype=x.dtype, shape=x.shape, needs_grad=False)

    def run(
        self,
        iters: int = int(1e6),
        iter_start: int = 0,
        report_interval: int = 100,
        callback: typing.Optional[typing.Callable] = None,
    ) -> None:
        @ti.kernel
        def gradient_descent(t: int):
            for i in range(self.x.shape[0]):
                self.m[i] = ti.math.mix(self.m[i], self.x.grad[i], self.beta_1)
                self.v[i] = ti.math.mix(self.v[i], self.x.grad[i] ** 2, self.beta_2)
                m_hat = self.m[i] / (1.0 - ti.pow(self.beta_1, t + 1))
                v_hat = self.v[i] / (1.0 - ti.pow(self.beta_2, t + 1))
                self.x[i] -= self.eta * m_hat / (ti.sqrt(v_hat) + self.epsilon)

        for i in range(iter_start, iters):
            self.loss[None] = 0
            with ti.ad.Tape(loss=self.loss):
                self.loss_fn()
            if report_interval and i % report_interval == 0:
                print(f"iter = {i},", f"loss = {self.loss[None]}")
            if callback:
                if callback(i):
                    break
            gradient_descent(i)
        else:
            print(f"iter = {iters},", f"loss = {self.loss[None]}")
            if callback:
                callback(iters)
