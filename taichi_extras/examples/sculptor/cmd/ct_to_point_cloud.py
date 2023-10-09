import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Annotated

import cv2 as cv
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import trimesh.transformations
import typer
from numpy.typing import NDArray
from trimesh import PointCloud


class Artifact(str, Enum):
    RAW = "raw"
    THRESHOLD = "threshold"
    OPENING = "opening"
    BACKGROUND = "background"
    COMPONENT = "component"
    EDGE = "edge"


def to_uint8(data: NDArray) -> NDArray[np.uint8]:
    img: NDArray = np.interp(data, (data.min(), data.max()), (0, 255))
    return img.astype(np.uint8)


def img_save(
    data: NDArray, id: int, artifact_path: Path = Path.cwd() / "artifact"
) -> None:
    os.makedirs(artifact_path, exist_ok=True)
    plt.imsave(str(artifact_path / f"{id:03d}.png"), data)


def edge_detection(
    id: int,
    data: NDArray,
    artifact_path: Path = Path.cwd() / "artifact",
    artifact_type: Sequence[Artifact] = [],
) -> NDArray:
    if Artifact.RAW in artifact_type:
        img_save(data=data, id=id, artifact_path=artifact_path / "raw")
    threshold, data = cv.threshold(
        src=data, thresh=0.0, maxval=255.0, type=cv.THRESH_BINARY_INV
    )
    if Artifact.THRESHOLD in artifact_type:
        img_save(data=data, id=id, artifact_path=artifact_path / "threshold")
    data = cv.morphologyEx(src=data, op=cv.MORPH_OPEN, kernel=np.ones(shape=(3, 3)))
    if Artifact.OPENING in artifact_type:
        img_save(data=data, id=id, artifact_path=artifact_path / "opening")
    num_components, labels, stats, centroids = cv.connectedComponentsWithStats(
        to_uint8(data)
    )
    background_label: int = np.argmax(
        stats[:, cv.CC_STAT_WIDTH] * stats[:, cv.CC_STAT_HEIGHT]  # type: ignore
    )
    data[labels == background_label] = False
    data[labels != background_label] = True
    if Artifact.COMPONENT in artifact_type:
        img_save(data=data, id=id, artifact_path=artifact_path / "background")
    num_components, labels, stats, centroids = cv.connectedComponentsWithStats(
        to_uint8(data)
    )
    for i in range(num_components):
        if centroids[i, 0] > 0.7 * data.shape[0]:
            data[labels == i] = 0
    if Artifact.COMPONENT in artifact_type:
        img_save(data=data, id=id, artifact_path=artifact_path / "component")
    # data = cv.Canny(image=to_uint8(data), threshold1=64.0, threshold2=128.0)
    # if Artifact.EDGE in artifact_type:
    #     img_save(data=data, id=id, artifact_path=artifact_path / "edge")
    return data


def main(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_path: Annotated[
        Path, typer.Argument(dir_okay=False, readable=False, writable=True)
    ],
    *,
    artifact_path: Annotated[
        Path, typer.Option(file_okay=False, readable=False, writable=True)
    ] = Path("artifact"),
    artifact_type: Annotated[list[Artifact], typer.Option(case_sensitive=False)] = [],
) -> None:
    data, header = nrrd.read(str(input_path))
    edge: NDArray
    with ProcessPoolExecutor() as executor:
        results: Sequence[NDArray] = list(
            executor.map(
                edge_detection,
                range(data.shape[-1]),
                np.moveaxis(data, source=-1, destination=0),
                [artifact_path] * data.shape[-1],
                [artifact_type] * data.shape[-1],
            )
        )
        edge = np.stack(results, axis=-1)
    vertices: NDArray = np.transpose(np.stack(np.where(edge), axis=0)).astype(float)
    space_origin: NDArray = header["space origin"]
    space_directions: NDArray = header["space directions"]
    vertices = vertices @ space_directions + space_origin
    import trimesh.voxel.ops

    print(header["space directions"])
    print(np.diag(header["space directions"]))
    print(np.diag(header["space directions"]).min())
    mesh = trimesh.voxel.ops.points_to_marching_cubes(
        vertices, pitch=np.diag(header["space directions"]).max()
    )
    mesh.export("cubes.ply")
    # point_cloud: PointCloud = PointCloud(vertices=vertices)
    # point_cloud.apply_transform(
    #     trimesh.transformations.rotation_matrix(angle=np.pi, direction=[0, 0, 1])
    #     @ trimesh.transformations.rotation_matrix(angle=-np.pi / 2, direction=[1, 0, 0])
    # )
    # point_cloud.apply_translation(-point_cloud.centroid)
    # point_cloud.export(output_path)


if __name__ == "__main__":
    typer.run(main)
