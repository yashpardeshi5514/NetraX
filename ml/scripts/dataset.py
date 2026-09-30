from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# NetraX — RFMiD Dataset Pipeline
# ============================================================

IMAGE_SIZE = 384
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# RFMiD disease labels — 45 classes.
LABEL_COLUMNS = [
    "DR", "ARMD", "MH", "DN", "MYA",
    "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH",
    "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC",
    "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD",
    "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]

NUM_CLASSES = len(LABEL_COLUMNS)

# Dataset is stored outside the project so it does not get committed.
DATASET_ROOT = Path(r"D:\Datasets\RFMiD")

SPLIT_CONFIG = {
    "train": {
        "images": DATASET_ROOT / "Training_Set" / "Training",
        "labels": DATASET_ROOT / "Training_Set" / "RFMiD_Training_Labels.csv",
    },
    "validation": {
        "images": DATASET_ROOT / "Evaluation_Set" / "Validation",
        "labels": DATASET_ROOT / "Evaluation_Set" / "RFMiD_Validation_Labels.csv",
    },
    "test": {
        "images": DATASET_ROOT / "Test_Set" / "Test",
        "labels": DATASET_ROOT / "Test_Set" / "RFMiD_Testing_Labels.csv",
    },
}


def build_dataframe(split_name):
    """
    Build a dataframe containing:
        image_path + 45 binary disease labels.

    The Disease_Risk column is intentionally not used as a model
    target because NetraX is being trained as a 45-label disease
    classifier.
    """
    if split_name not in SPLIT_CONFIG:
        raise ValueError(
            f"Unknown split '{split_name}'. "
            f"Expected one of: {list(SPLIT_CONFIG)}"
        )

    config = SPLIT_CONFIG[split_name]

    if not config["images"].exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{config['images']}"
        )

    if not config["labels"].exists():
        raise FileNotFoundError(
            f"Label CSV not found:\n{config['labels']}"
        )

    df = pd.read_csv(config["labels"])

    required_columns = ["ID", "Disease_Risk"] + LABEL_COLUMNS
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {config['labels']}:\n{missing}"
        )

    df = df.copy()

    # RFMiD IDs are used as the PNG filenames.
    df["image_path"] = df["ID"].apply(
        lambda image_id: str(config["images"] / f"{image_id}.png")
    )

    missing_images = df.loc[
        ~df["image_path"].map(Path).map(Path.exists),
        ["ID", "image_path"],
    ]

    if not missing_images.empty:
        raise FileNotFoundError(
            f"{len(missing_images)} image files are missing in {split_name}."
        )

    # Ensure labels are numeric binary values.
    df[LABEL_COLUMNS] = (
        df[LABEL_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .astype(np.float32)
    )

    return df[["ID", "image_path"] + LABEL_COLUMNS]


def load_image(path, labels):
    """
    Load PNG and resize using aspect-ratio-preserving padding.

    Important:
    EfficientNet's Keras implementation performs its own input
    rescaling, so this pipeline intentionally keeps pixel values
    in the [0, 255] float32 range.
    """
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_png(image_bytes, channels=3)

    image = tf.image.resize_with_pad(
        image,
        target_height=IMAGE_SIZE,
        target_width=IMAGE_SIZE,
        method=tf.image.ResizeMethod.BILINEAR,
    )

    image = tf.cast(image, tf.float32)

    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    labels.set_shape((NUM_CLASSES,))

    return image, labels


def augment(image, labels):
    """
    Conservative augmentation for retinal fundus images.

    No vertical flipping or aggressive random cropping is used,
    because those operations can unnecessarily alter retinal
    anatomy or remove diagnostically relevant regions.
    """
    image = tf.image.random_brightness(
        image,
        max_delta=12.0,
    )

    image = tf.image.random_contrast(
        image,
        lower=0.90,
        upper=1.10,
    )

    # Horizontal flip is safe with the current disease labels because
    # RFMiD labels do not encode left/right eye laterality.
    image = tf.image.random_flip_left_right(image)

    image = tf.clip_by_value(image, 0.0, 255.0)

    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))

    return image, labels


def create_dataset(
    split_name,
    batch_size=BATCH_SIZE,
    training=False,
):
    """
    Create a tf.data.Dataset for train/validation/test.
    """
    df = build_dataframe(split_name)

    paths = df["image_path"].values
    labels = df[LABEL_COLUMNS].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        dataset = dataset.map(
            augment,
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def calculate_positive_counts(split_name="train"):
    """
    Calculate positive samples for each disease directly from CSV.

    This avoids manually hard-coding class counts in the training
    pipeline.
    """
    df = build_dataframe(split_name)

    return df[LABEL_COLUMNS].sum(axis=0).astype(np.int64)


def calculate_class_weights(
    split_name="train",
    max_weight=10.0,
):
    """
    Calculate capped positive-class weights.

    weight = negatives / positives

    A maximum of 10x is used to prevent extremely rare RFMiD
    labels from dominating optimization.
    """
    df = build_dataframe(split_name)

    positive_counts = df[LABEL_COLUMNS].sum(axis=0).to_numpy(
        dtype=np.float32
    )

    total_samples = float(len(df))
    negative_counts = total_samples - positive_counts

    weights = np.divide(
        negative_counts,
        positive_counts,
        out=np.ones_like(positive_counts),
        where=positive_counts > 0,
    )

    weights = np.clip(
        weights,
        1.0,
        max_weight,
    )

    return pd.Series(
        weights,
        index=LABEL_COLUMNS,
        name="positive_weight",
    )


def print_dataset_summary():
    """
    Print a compact verification of all three RFMiD splits.
    """
    print("=" * 70)
    print("NETRAX — RFMiD DATASET PIPELINE")
    print("=" * 70)

    for split in ("train", "validation", "test"):
        df = build_dataframe(split)

        print(f"\n{split.upper()}")
        print("-" * 70)
        print("Images:", len(df))
        print("Labels:", NUM_CLASSES)
        print(
            "Positive labels:",
            int(df[LABEL_COLUMNS].to_numpy().sum()),
        )

    print("\nLabel count:", NUM_CLASSES)
    print("Image size:", f"{IMAGE_SIZE}x{IMAGE_SIZE}")
    print("Batch size:", BATCH_SIZE)


if __name__ == "__main__":
    print_dataset_summary()

    print("\nCreating training batch...")
    train_dataset = create_dataset(
        "train",
        training=True,
    )

    images, labels = next(iter(train_dataset))

    print("Training image batch shape:", images.shape)
    print("Training label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    print("Image minimum:", float(tf.reduce_min(images)))
    print("Image maximum:", float(tf.reduce_max(images)))

    print("\nChecking validation batch...")
    validation_dataset = create_dataset(
        "validation",
        training=False,
    )

    val_images, val_labels = next(iter(validation_dataset))

    print("Validation image batch shape:", val_images.shape)
    print("Validation label batch shape:", val_labels.shape)
    print("Validation image minimum:", float(tf.reduce_min(val_images)))
    print("Validation image maximum:", float(tf.reduce_max(val_images)))

    print("\nCalculating class weights directly from training CSV...")
    class_weights = calculate_class_weights("train")

    print(
        "\nWeight range:",
        f"{class_weights.min():.2f} - {class_weights.max():.2f}",
    )

    print("\nFirst 10 class weights:")
    print(class_weights.head(10).to_string())

    print("\nDataset pipeline test completed successfully.")
