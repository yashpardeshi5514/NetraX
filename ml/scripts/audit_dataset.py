from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"D:\Datasets\RFMiD")

SPLITS = {
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


# ============================================================
# HELPERS
# ============================================================

def find_images(folder):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    return [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]


def inspect_images(image_paths):
    dimensions = Counter()
    formats = Counter()
    corrupt = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                dimensions[img.size] += 1
                formats[img.format] += 1

                # Force PIL to verify the image
                img.verify()

        except Exception:
            corrupt.append(str(image_path))

    return dimensions, formats, corrupt


# ============================================================
# DATASET AUDIT
# ============================================================

print("\n" + "=" * 70)
print("NETRAX — RFMiD DATASET AUDIT")
print("=" * 70)


all_label_counts = Counter()


for split_name, paths in SPLITS.items():

    print("\n" + "-" * 70)
    print(f"{split_name.upper()} SET")
    print("-" * 70)

    image_dir = paths["images"]
    label_file = paths["labels"]

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    print(f"\nImages : {image_dir}")
    print(f"Labels : {label_file}")

    if not image_dir.exists():
        print("❌ Image directory does not exist")
        continue

    if not label_file.exists():
        print("❌ Label CSV does not exist")
        continue

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    df = pd.read_csv(label_file)

    print(f"\nCSV rows: {len(df)}")
    print(f"CSV columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Identify disease columns
    # --------------------------------------------------------

    disease_columns = [
        column
        for column in df.columns
        if column not in ["ID", "Disease_Risk"]
    ]

    print(f"Disease labels: {len(disease_columns)}")

    print("\nDisease labels:")
    print(", ".join(disease_columns))

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    image_paths = find_images(image_dir)

    print(f"\nImages found: {len(image_paths)}")

    # --------------------------------------------------------
    # Compare image IDs and CSV IDs
    # --------------------------------------------------------

    image_ids = set()

    for image_path in image_paths:

        try:
            image_id = int(image_path.stem)
            image_ids.add(image_id)

        except ValueError:
            pass

    csv_ids = set(df["ID"].astype(int))

    missing_images = csv_ids - image_ids
    missing_labels = image_ids - csv_ids

    print(f"\nMissing image files: {len(missing_images)}")
    print(f"Images without labels: {len(missing_labels)}")

    if missing_images:
        print("Missing image IDs:")
        print(sorted(missing_images)[:20])

    if missing_labels:
        print("Images without labels:")
        print(sorted(missing_labels)[:20])

    # --------------------------------------------------------
    # Disease distribution
    # --------------------------------------------------------

    print("\nDisease distribution:")

    split_label_counts = {}

    for disease in disease_columns:

        count = int(df[disease].sum())

        split_label_counts[disease] = count
        all_label_counts[disease] += count

        print(f"{disease:8} : {count}")

    # --------------------------------------------------------
    # Disease risk
    # --------------------------------------------------------

    print("\nDisease_Risk distribution:")

    risk_counts = df["Disease_Risk"].value_counts().sort_index()

    for value, count in risk_counts.items():
        print(f"Risk {value}: {count}")

    # --------------------------------------------------------
    # Multi-label cases
    # --------------------------------------------------------

    disease_matrix = df[disease_columns]

    labels_per_image = disease_matrix.sum(axis=1)

    print("\nLabels per image:")

    print(
        f"Minimum labels : {labels_per_image.min()}"
    )

    print(
        f"Maximum labels : {labels_per_image.max()}"
    )

    print(
        f"Average labels : {labels_per_image.mean():.3f}"
    )

    multi_label_count = int((labels_per_image > 1).sum())

    print(
        f"Multi-label images: {multi_label_count}"
    )

    # --------------------------------------------------------
    # Image inspection
    # --------------------------------------------------------

    print("\nInspecting images...")

    dimensions, formats, corrupt = inspect_images(image_paths)

    print("\nImage formats:")

    for image_format, count in formats.items():
        print(f"{image_format}: {count}")

    print("\nImage dimensions:")

    for dimension, count in dimensions.most_common():
        print(f"{dimension}: {count}")

    print(f"\nCorrupt images: {len(corrupt)}")

    if corrupt:
        print("\nCorrupt files:")

        for file in corrupt[:20]:
            print(file)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL DISEASE COUNTS")
print("=" * 70)

for disease, count in all_label_counts.most_common():
    print(f"{disease:8} : {count}")


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)