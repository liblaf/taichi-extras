import typing

import taichi as ti


class GradientDescent:
    loss_fn: typing.Callable
    variables: list
