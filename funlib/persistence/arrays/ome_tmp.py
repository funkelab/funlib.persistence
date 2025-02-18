from typing import Literal

import numpy as np
from iohub.ngff import TransformationMeta, Position, TiledPosition, TiledImageArray


def _get_all_transforms(
    node: Position | TiledImageArray | TiledPosition, image: str | Literal["*"]
) -> list[TransformationMeta]:
    """Get all transforms metadata
    for one image array or the whole FOV.

    Parameters
    ----------
    image : str | Literal["*"]
        Name of one image array (e.g. "0") to query,
        or "*" for the whole FOV

    Returns
    -------
    list[TransformationMeta]
        All transforms applicable to this image or FOV.
    """
    transforms: list[TransformationMeta] = (
        [t for t in node.metadata.multiscales[0].coordinate_transformations]
        if node.metadata.multiscales[0].coordinate_transformations is not None
        else []
    )
    if image != "*" and image in node:
        for i, dataset_meta in enumerate(node.metadata.multiscales[0].datasets):
            if dataset_meta.path == image:
                transforms.extend(
                    node.metadata.multiscales[0].datasets[i].coordinate_transformations
                )
    elif image != "*":
        raise ValueError(f"Key {image} not recognized.")
    return transforms


def get_effective_scale(
    node,
    image: str | Literal["*"],
) -> list[float]:
    """Get the effective coordinate scale metadata
    for one image array or the whole FOV.

    Parameters
    ----------
    image : str | Literal["*"]
        Name of one image array (e.g. "0") to query,
        or "*" for the whole FOV

    Returns
    -------
    list[float]
        A list of floats representing the total scale
        for the image or FOV for each axis.
    """
    transforms = _get_all_transforms(node, image)

    full_scale = np.ones(len(node.axes), dtype=float)
    for transform in transforms:
        if transform.type == "scale":
            full_scale *= np.array(transform.scale)

    return [float(x) for x in full_scale]


def get_effective_translation(
    node,
    image: str | Literal["*"],
) -> TransformationMeta:
    """Get the effective coordinate translation metadata
    for one image array or the whole FOV.

    Parameters
    ----------
    image : str | Literal["*"]
        Name of one image array (e.g. "0") to query,
        or "*" for the whole FOV

    Returns
    -------
    list[float]
        A list of floats representing the total translation
        for the image or FOV for each axis.
    """
    transforms = _get_all_transforms(node, image)
    full_translation = np.zeros(len(node.axes), dtype=float)
    for transform in transforms:
        if transform.type == "translation":
            full_translation += np.array(transform.translation)

    return [float(x) for x in full_translation]
