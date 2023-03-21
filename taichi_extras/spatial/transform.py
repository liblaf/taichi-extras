import numpy as np
import taichi as ti


def get_matrix(params: np.ndarray) -> tuple[ti.math.mat3, ti.math.vec3]:
    transform: ti.Matrix = ti.Matrix(np.reshape(params[:9], newshape=(3, 3)))
    displacement: ti.Vector = ti.Vector(np.reshape(params[-3:], newshape=(3,)))
    return transform, displacement


@ti.func
def transform_point(
    transform: ti.template(), displacement: ti.template(), point: ti.math.vec3
) -> ti.math.vec3:
    return transform @ point + displacement


@ti.func
def inverse_transform_point(
    transform: ti.template(), displacement: ti.template(), point: ti.math.vec3
) -> ti.math.vec3:
    return ti.math.inverse(transform) @ (point - displacement)


def transform_mesh(
    transform: ti.math.mat3,
    displacement: ti.math.vec3,
    input_points: ti.MatrixField,
    output_points: ti.MatrixField,
):
    @ti.kernel
    def run():
        for i in range(input_points.shape[0]):
            output_points[i] = transform_point(
                transform=transform, displacement=displacement, point=input_points[i]
            )

    run()


def inverse_transform_mesh(
    transform: ti.math.mat3,
    displacement: ti.math.vec3,
    input_points: ti.MatrixField,
    output_points: ti.MatrixField,
):
    @ti.kernel
    def run():
        for i in range(input_points.shape[0]):
            output_points[i] = inverse_transform_point(
                transform=transform, displacement=displacement, point=input_points[i]
            )

    run()
