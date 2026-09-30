from pathlib import Path

from preprocessing import preprocess_image


DATASET_ROOT = Path(r"D:\Datasets\RFMiD")

IMAGE_DIR = (
    DATASET_ROOT
    / "Training_Set"
    / "Training"
)


images = list(
    IMAGE_DIR.glob("*.png")
)


print(f"Images found: {len(images)}")


for image_path in images[:5]:

    image = preprocess_image(image_path)

    print(
        f"{image_path.name} -> "
        f"shape={image.shape}, "
        f"dtype={image.dtype}, "
        f"min={image.min():.1f}, "
        f"max={image.max():.1f}"
    )