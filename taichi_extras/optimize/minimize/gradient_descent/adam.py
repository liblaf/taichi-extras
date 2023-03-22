import typing

import taichi as ti

from taichi_extras.utils.field.like import field_like


class Adam:
    loss_fn: typing.Callable[[], None]

    beta_1: float
    beta_2: float
    epsilon: float
    eta: float

    m: tuple[ti.ScalarField | ti.MatrixField, ...]
    v: tuple[ti.ScalarField | ti.MatrixField, ...]
    x: tuple[ti.ScalarField | ti.MatrixField, ...]
    loss: ti.ScalarField

    def __init__(
        self,
        loss_fn: typing.Callable,
        loss: ti.ScalarField,
        x: tuple[ti.ScalarField | ti.MatrixField, ...],
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

        self.m = tuple(field_like(_x, needs_grad=False) for _x in x)
        self.v = tuple(field_like(_x, needs_grad=False) for _x in x)

    def run(
        self,
        iters: int = int(1e6),
        iter_start: int = 0,
        report_interval: int = 100,
        callback: typing.Optional[typing.Callable] = None,
    ) -> None:
        gradient_descent_functions: list[typing.Callable[[int], None]] = list()
        for i in range(len(self.x)):

            @ti.kernel
            def run(t: int):
                for I in ti.grouped(ti.ndrange(*(self.x[i].shape))):
                    self.m[i][I] = ti.math.mix(
                        self.m[i][I], self.x[i].grad[I], self.beta_1
                    )
                    self.v[i][I] = ti.math.mix(
                        self.v[i][I], self.x[i].grad[I] ** 2, self.beta_2
                    )
                    m_hat = self.m[i][I] / (1.0 - ti.pow(self.beta_1, t + 1))
                    v_hat = self.v[i][I] / (1.0 - ti.pow(self.beta_2, t + 1))
                    self.x[i][I] -= self.eta * m_hat / (ti.sqrt(v_hat) + self.epsilon)

            gradient_descent_functions.append(run)

        def gradient_descent(t: int):
            for i in range(len(self.x)):
                gradient_descent_functions[i](t)

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
