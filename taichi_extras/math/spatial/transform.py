import numpy as np
import taichi as ti


def to_matrix(params: np.ndarray) -> tuple[ti.math.mat3, ti.math.vec3]:
    transform: ti.math.mat3 = ti.Matrix(np.reshape(params[:9], newshape=(3, 3)))
    displacement: ti.Vector = ti.Vector(np.reshape(params[-3:], newshape=(3,)))
    return transform, displacement


def to_numpy(transform: ti.math.mat3, displacement: ti.math.vec3) -> np.ndarray:
    return np.array(
        np.concatenate(
            [transform.to_numpy().flatten(), displacement.to_numpy().flatten()]
        )
    )


@ti.func
def transform_point(
    transform: ti.template(),  # type: ignore
    displacement: ti.template(),  # type: ignore
    point: ti.math.vec3,
) -> ti.math.vec3:
    return transform @ point + displacement


@ti.func
def inverse_transform_point(
    transform: ti.template(),  # type: ignore
    displacement: ti.template(),  # type: ignore
    point: ti.math.vec3,
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
        for I in ti.grouped(ti.ndrange(*(input_points.shape))):
            output_points[I] = transform_point(
                transform=transform, displacement=displacement, point=input_points[I]
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
        for I in ti.grouped(ti.ndrange(*(input_points.shape))):
            output_points[I] = inverse_transform_point(
                transform=transform, displacement=displacement, point=input_points[I]
            )

    run()
