from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMG_SIZE = 384


def load_image(image_path: str | Path) -> Image.Image:
    """
    Load a retinal fundus image and convert it to RGB.
    """

    image = Image.open(image_path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def resize_with_padding(
    image: Image.Image,
    size: int = IMG_SIZE
) -> Image.Image:
    """
    Resize while preserving aspect ratio and pad to a square.

    This avoids stretching the retinal image.
    """

    image = ImageOps.contain(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        (size, size),
        (0, 0, 0)
    )

    x = (size - image.width) // 2
    y = (size - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


def preprocess_image(
    image_path: str | Path,
    size: int = IMG_SIZE
) -> np.ndarray:
    """
    Complete inference preprocessing.

    Returns:
        Float32 NumPy array with shape:
        (size, size, 3)
    """

    image = load_image(image_path)

    image = resize_with_padding(
        image,
        size=size
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    return image