from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

from dataset import create_dataset, LABEL_COLUMNS


IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_CLASSES = 45

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "netrax_finetuned_best.keras"
EVAL_DIR = PROJECT_ROOT / "evaluation"

THRESHOLD_JSON = EVAL_DIR / "finetuned_per_label_thresholds.json"
THRESHOLD_CSV = EVAL_DIR / "finetuned_per_label_thresholds.csv"


def collect_data(dataset):
    images = []
    labels = []

    for x, y in dataset:
        images.append(x.numpy())
        labels.append(y.numpy())

    return (
        np.concatenate(images, axis=0),
        np.concatenate(labels, axis=0),
    )


def find_best_threshold(y_true, probabilities):
    candidates = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in candidates:
        predictions = (
            probabilities >= threshold
        ).astype(np.int32)

        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            predictions,
            average="binary",
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NetraX — Fine-Tuned Model Threshold Optimization")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found:\n{MODEL_PATH}"
        )

    # ---------------------------------------------------------
    # Validation dataset ONLY
    # ---------------------------------------------------------
    print("\nLoading validation dataset...")

    val_ds = create_dataset(
        "validation",
        batch_size=BATCH_SIZE,
        training=False,
    )

    print("Collecting validation data...")

    _, y_val = collect_data(val_ds)

    print("Loading fine-tuned model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Generating validation probabilities...")

    val_probabilities = model.predict(
        val_ds,
        verbose=1,
    )

    if val_probabilities.shape != y_val.shape:
        raise ValueError(
            f"Prediction shape {val_probabilities.shape} "
            f"does not match labels {y_val.shape}."
        )

    # ---------------------------------------------------------
    # Optimize one threshold per label
    # ---------------------------------------------------------
    results = {}

    for index, label in enumerate(LABEL_COLUMNS):
        y_true = y_val[:, index]
        probabilities = val_probabilities[:, index]

        positive_support = int(y_true.sum())
        negative_support = int(len(y_true) - positive_support)

        # Extremely small validation support makes threshold
        # optimization unstable. Keep the conservative default.
        if positive_support < 5 or negative_support < 5:
            threshold = 0.50
            validation_f1 = None
            status = "default_low_support"
        else:
            threshold, validation_f1 = find_best_threshold(
                y_true,
                probabilities,
            )
            status = "optimized"

        results[label] = {
            "threshold": float(threshold),
            "validation_f1": (
                float(validation_f1)
                if validation_f1 is not None
                else None
            ),
            "positive_support": positive_support,
            "negative_support": negative_support,
            "status": status,
        }

        f1_text = (
            f"{validation_f1:.4f}"
            if validation_f1 is not None
            else "N/A"
        )

        print(
            f"{label:6s} | "
            f"support={positive_support:3d} | "
            f"threshold={threshold:.2f} | "
            f"val F1={f1_text} | "
            f"{status}"
        )

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------
    with open(
        THRESHOLD_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------
    rows = []

    for label, values in results.items():
        rows.append(
            {
                "label": label,
                **values,
            }
        )

    pd.DataFrame(rows).to_csv(
        THRESHOLD_CSV,
        index=False,
    )

    optimized_count = sum(
        value["status"] == "optimized"
        for value in results.values()
    )

    default_count = len(results) - optimized_count

    print("\n" + "=" * 70)
    print("Threshold optimization complete")
    print("=" * 70)
    print(f"Total labels:       {len(results)}")
    print(f"Optimized labels:   {optimized_count}")
    print(f"Default labels:     {default_count}")
    print(f"JSON:               {THRESHOLD_JSON}")
    print(f"CSV:                {THRESHOLD_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()