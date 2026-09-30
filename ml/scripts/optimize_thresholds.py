import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import f1_score

from dataset import (
    LABEL_COLUMNS,
    NUM_CLASSES,
    BATCH_SIZE,
    create_dataset,
)


# ============================================================
# NetraX — Per-Disease Threshold Optimization
# ============================================================

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "netrax_best.keras",
    )
)

OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "evaluation",
    )
)

THRESHOLD_JSON = os.path.join(
    OUTPUT_DIR,
    "per_label_thresholds.json",
)

THRESHOLD_CSV = os.path.join(
    OUTPUT_DIR,
    "per_label_thresholds.csv",
)

# Candidate thresholds.
THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.05,
    dtype=np.float32,
)

# Very rare validation labels should not have their threshold
# aggressively optimized because the estimate is unstable.
MIN_POSITIVE_SUPPORT = 5
MIN_NEGATIVE_SUPPORT = 5


def collect_predictions(model, dataset):
    """Collect validation labels and probabilities."""
    y_true = []
    y_prob = []

    for images, labels in dataset:
        probabilities = model(
            images,
            training=False,
        ).numpy()

        y_true.append(labels.numpy())
        y_prob.append(probabilities)

    return (
        np.concatenate(y_true, axis=0),
        np.concatenate(y_prob, axis=0),
    )


def optimize_label_threshold(y_true, y_prob):
    """
    Select the threshold maximizing F1 for one disease.

    For very rare labels, return 0.50 instead of overfitting a
    threshold to only a handful of validation positives.
    """
    positive_support = int(np.sum(y_true))
    negative_support = int(len(y_true) - positive_support)

    if (
        positive_support < MIN_POSITIVE_SUPPORT
        or negative_support < MIN_NEGATIVE_SUPPORT
    ):
        return {
            "threshold": 0.50,
            "f1": np.nan,
            "support": positive_support,
            "optimized": False,
        }

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in THRESHOLDS:
        predictions = (
            y_prob >= threshold
        ).astype(np.int32)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "support": positive_support,
        "optimized": True,
    }


def optimize_thresholds():
    print("=" * 70)
    print("NETRAX — PER-DISEASE THRESHOLD OPTIMIZATION")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("\nLoading model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("\nLoading validation dataset...")

    validation_dataset = create_dataset(
        "validation",
        batch_size=BATCH_SIZE,
        training=False,
    )

    print("Collecting validation predictions...")

    y_true, y_prob = collect_predictions(
        model,
        validation_dataset,
    )

    if y_true.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} labels, "
            f"got {y_true.shape[1]}"
        )

    results = []
    thresholds = {}

    for index, label in enumerate(LABEL_COLUMNS):

        result = optimize_label_threshold(
            y_true[:, index],
            y_prob[:, index],
        )

        thresholds[label] = result["threshold"]

        results.append(
            {
                "label": label,
                "support": result["support"],
                "threshold": result["threshold"],
                "validation_f1": result["f1"],
                "optimized": result["optimized"],
            }
        )

    report = pd.DataFrame(results)

    report.to_csv(
        THRESHOLD_CSV,
        index=False,
    )

    with open(
        THRESHOLD_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            thresholds,
            file,
            indent=2,
        )

    print("\n" + "-" * 70)
    print("PER-LABEL THRESHOLDS")
    print("-" * 70)

    print(
        report.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    optimized_count = int(
        report["optimized"].sum()
    )

    default_count = len(report) - optimized_count

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(
        "Labels optimized:",
        optimized_count,
        "/",
        NUM_CLASSES,
    )

    print(
        "Labels using default 0.50:",
        default_count,
        "/",
        NUM_CLASSES,
    )

    print(
        "\nSaved JSON:",
        THRESHOLD_JSON,
    )

    print(
        "Saved CSV:",
        THRESHOLD_CSV,
    )

    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    optimize_thresholds()
