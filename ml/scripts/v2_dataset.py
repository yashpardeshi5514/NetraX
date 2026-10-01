from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


IMAGE_SIZE = 384
NUM_CLASSES = 45
BATCH_SIZE = 16


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "dataset"
    / "v2_manifests"
)


LABELS = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS",
    "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
    "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP",
    "CWS", "CB", "ODPM", "PRH", "MNF", "HR", "CRAO", "TD", "CME",
    "PTCR", "CF", "VH", "MCA", "VS", "BRAO", "PLQ", "HPED", "CL"
]


LABEL_INDEX = {
    label: i
    for i, label in enumerate(LABELS)
}


def resolve_image_path(row):
    source = row["source"]
    split = row["split"]
    filename = row["filename"]

    # --------------------------------------------------------
    # RFMiD
    # --------------------------------------------------------

    if source == "RFMiD":

        rfmid_root = Path(
            r"D:\Datasets\RFMiD"
        )

        if split == "train":
            return (
                rfmid_root
                / "Training_Set"
                / "Training"
                / filename
            )

        if split == "validation":
            return (
                rfmid_root
                / "Evaluation_Set"
                / "Validation"
                / filename
            )

        if split == "test":
            return (
                rfmid_root
                / "Test_Set"
                / "Test"
                / filename
            )

    # --------------------------------------------------------
    # ODIR
    # --------------------------------------------------------

    if source == "ODIR":

        return (
            PROJECT_ROOT
            / "ml"
            / "dataset"
            / "odir_v2"
            / split
            / filename
        )

    raise ValueError(
        f"Unknown source={source}, split={split}"
    )


def load_manifest(split, source=None):

    path = (
        MANIFEST_ROOT
        / f"{split}_manifest_fixed.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    df = pd.read_csv(path)

    if source is not None:

        df = df[
            df["source"] == source
        ].copy()

    if df.empty:

        raise ValueError(
            f"No records found for "
            f"split={split}, source={source}"
        )

    return df.reset_index(
        drop=True
    )


def prepare_dataframe(df):

    df = df.copy()

    paths = []

    for _, row in df.iterrows():

        path = resolve_image_path(
            row
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        paths.append(
            str(path)
        )

    df["image_path"] = paths

    return df


def preprocess_image(path):

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize_with_pad(
        image,
        IMAGE_SIZE,
        IMAGE_SIZE,
        method="bilinear"
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.clip_by_value(
        image,
        0.0,
        255.0
    )

    return image


def augment_image(image):

    # Conservative augmentation for
    # retinal fundus images.

    image = tf.image.random_brightness(
        image,
        max_delta=12.0
    )

    image = tf.image.random_contrast(
        image,
        lower=0.90,
        upper=1.10
    )

    image = tf.image.random_flip_left_right(
        image
    )

    image = tf.clip_by_value(
        image,
        0.0,
        255.0
    )

    return image


def dataframe_to_arrays(df):

    paths = (
        df["image_path"]
        .astype(str)
        .values
    )

    targets = (
        df[LABELS]
        .astype(np.float32)
        .values
    )

    known_columns = [
        f"{label}_known"
        for label in LABELS
    ]

    masks = (
        df[known_columns]
        .astype(np.float32)
        .values
    )

    return (
        paths,
        targets,
        masks
    )


def create_dataset(
    split,
    source=None,
    batch_size=BATCH_SIZE,
    training=False,
    shuffle=True
):

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    df = load_manifest(
        split=split,
        source=source
    )

    # --------------------------------------------------------
    # Resolve image paths
    # --------------------------------------------------------

    df = prepare_dataframe(
        df
    )

    # --------------------------------------------------------
    # Convert dataframe to arrays
    # --------------------------------------------------------

    paths, targets, masks = (
        dataframe_to_arrays(df)
    )

    # --------------------------------------------------------
    # Create TensorFlow dataset
    # --------------------------------------------------------

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            targets,
            masks
        )
    )

    # --------------------------------------------------------
    # Shuffle only when training
    # --------------------------------------------------------

    if training and shuffle:

        dataset = dataset.shuffle(
            buffer_size=min(
                len(df),
                2048
            ),
            reshuffle_each_iteration=True
        )

    # --------------------------------------------------------
    # Load and preprocess each image
    # --------------------------------------------------------

    def load_example(
        path,
        target,
        mask
    ):

        image = preprocess_image(
            path
        )

        if training:

            image = augment_image(
                image
            )

        return (
            image,
            {
                "targets": target,
                "mask": mask,
            }
        )

    dataset = dataset.map(
        load_example,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False
    )

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    # IMPORTANT:
    # create_dataset returns BOTH:
    #
    #     dataset
    #     dataframe
    #
    return dataset, df


def inspect_dataset(
    split,
    source=None,
    batch_size=BATCH_SIZE
):

    dataset, df = create_dataset(
        split=split,
        source=source,
        batch_size=batch_size,
        training=False,
        shuffle=False
    )

    images, labels = next(
        iter(dataset)
    )

    targets = labels["targets"]
    masks = labels["mask"]

    print("=" * 70)
    print("V2 DATASET PIPELINE TEST")
    print("=" * 70)

    print()
    print("Split:", split)
    print("Source:", source)
    print("Records:", len(df))

    print()
    print("Image tensor:")
    print("Shape:", images.shape)
    print("Dtype:", images.dtype)

    print(
        "Range:",
        float(
            tf.reduce_min(
                images
            ).numpy()
        ),
        "to",
        float(
            tf.reduce_max(
                images
            ).numpy()
        )
    )

    print()
    print("Target tensor:")
    print("Shape:", targets.shape)
    print("Dtype:", targets.dtype)

    print()
    print("Mask tensor:")
    print("Shape:", masks.shape)
    print("Dtype:", masks.dtype)

    print()
    print(
        "Known label values:",
        np.unique(
            masks.numpy()
        )
    )

    print()
    print("Known labels in first batch:")

    known_counts = (
        tf.reduce_sum(
            masks,
            axis=0
        ).numpy()
    )

    for i, label in enumerate(LABELS):

        if known_counts[i] > 0:

            print(
                "{:<6} {}".format(
                    label,
                    int(
                        known_counts[i]
                    )
                )
            )

    print()
    print("Positive labels in first batch:")

    positive_counts = (
        tf.reduce_sum(
            targets,
            axis=0
        ).numpy()
    )

    for i, label in enumerate(LABELS):

        if positive_counts[i] > 0:

            print(
                "{:<6} {}".format(
                    label,
                    int(
                        positive_counts[i]
                    )
                )
            )

    print()
    print("Pipeline test passed.")
    print("=" * 70)


if __name__ == "__main__":

    print()
    print(
        "Testing combined training pipeline..."
    )

    inspect_dataset(
        split="train",
        source=None
    )

    print()
    print(
        "Testing RFMiD-only validation..."
    )

    inspect_dataset(
        split="validation",
        source="RFMiD"
    )

    print()
    print(
        "Testing ODIR-only validation..."
    )

    inspect_dataset(
        split="validation",
        source="ODIR"
    )