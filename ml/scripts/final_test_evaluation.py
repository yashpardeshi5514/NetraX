from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataset import create_dataset, LABEL_COLUMNS


# ============================================================
# NetraX — FINAL TEST EVALUATION
# Fine-tuned model + validation-derived per-label thresholds
# ============================================================

BATCH_SIZE = 16
NUM_CLASSES = 45

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "netrax_finetuned_best.keras"
THRESHOLD_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "finetuned_per_label_thresholds.json"
)

EVAL_DIR = PROJECT_ROOT / "evaluation"

METRICS_CSV = EVAL_DIR / "final_test_metrics.csv"
SUMMARY_JSON = EVAL_DIR / "final_test_summary.json"


def collect_labels(dataset):
    labels = []

    for _, y in dataset:
        labels.append(y.numpy())

    return np.concatenate(labels, axis=0).astype(np.float32)


def calculate_per_label_metrics(y_true, probabilities, thresholds):
    predictions = (
        probabilities >= np.asarray(thresholds)[None, :]
    ).astype(np.int32)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    micro_precision, micro_recall, micro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="micro",
            zero_division=0,
        )
    )

    per_precision, per_recall, per_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average=None,
            zero_division=0,
        )
    )

    rows = []

    for i, label in enumerate(LABEL_COLUMNS):
        true_label = y_true[:, i]
        prob_label = probabilities[:, i]

        support = int(true_label.sum())

        if np.unique(true_label).size < 2:
            roc_auc = None
            pr_auc = None
        else:
            roc_auc = float(
                roc_auc_score(true_label, prob_label)
            )
            pr_auc = float(
                average_precision_score(true_label, prob_label)
            )

        rows.append(
            {
                "label": label,
                "support": support,
                "threshold": float(thresholds[i]),
                "precision": float(per_precision[i]),
                "recall": float(per_recall[i]),
                "f1": float(per_f1[i]),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    summary = {
        "test_samples": int(len(y_true)),
        "num_labels": NUM_CLASSES,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
    }

    return summary, rows


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NetraX — FINAL TEST EVALUATION")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found:\n{MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found:\n{THRESHOLD_PATH}"
        )

    # ---------------------------------------------------------
    # Load thresholds created ONLY from validation data
    # ---------------------------------------------------------
    print("\nLoading validation-derived thresholds...")

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        threshold_data = json.load(file)

    thresholds = []

    for label in LABEL_COLUMNS:
        if label not in threshold_data:
            raise ValueError(
                f"Missing threshold for label: {label}"
            )

        thresholds.append(
            float(threshold_data[label]["threshold"])
        )

    thresholds = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    if len(thresholds) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} thresholds, "
            f"got {len(thresholds)}."
        )

    print(
        f"Loaded {len(thresholds)} label thresholds."
    )

    # ---------------------------------------------------------
    # Load TEST dataset ONLY
    # ---------------------------------------------------------
    print("\nLoading untouched test dataset...")

    test_ds = create_dataset(
        "test",
        batch_size=BATCH_SIZE,
        training=False,
    )

    print("Collecting test labels...")

    y_test = collect_labels(test_ds)

    if y_test.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} test labels, "
            f"got {y_test.shape[1]}."
        )

    # ---------------------------------------------------------
    # Load final candidate model
    # ---------------------------------------------------------
    print("\nLoading fine-tuned model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} model outputs, "
            f"got {model.output_shape[-1]}."
        )

    # ---------------------------------------------------------
    # Generate TEST predictions
    # ---------------------------------------------------------
    print("\nGenerating test predictions...")

    probabilities = model.predict(
        test_ds,
        verbose=1,
    )

    if probabilities.shape != y_test.shape:
        raise ValueError(
            f"Prediction shape {probabilities.shape} "
            f"does not match test labels {y_test.shape}."
        )

    # ---------------------------------------------------------
    # Calculate FINAL metrics
    # ---------------------------------------------------------
    summary, rows = calculate_per_label_metrics(
        y_test,
        probabilities,
        thresholds,
    )

    metrics_df = pd.DataFrame(rows)

    metrics_df.to_csv(
        METRICS_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # Additional aggregate information
    # ---------------------------------------------------------
    valid_roc = metrics_df["roc_auc"].dropna()
    valid_pr = metrics_df["pr_auc"].dropna()

    summary["mean_per_label_roc_auc"] = (
        float(valid_roc.mean())
        if len(valid_roc)
        else None
    )

    summary["mean_per_label_pr_auc"] = (
        float(valid_pr.mean())
        if len(valid_pr)
        else None
    )

    summary["labels_with_test_positive_support"] = int(
        (metrics_df["support"] > 0).sum()
    )

    summary["labels_with_test_zero_support"] = int(
        (metrics_df["support"] == 0).sum()
    )

    summary["threshold_source"] = (
        "Validation set only"
    )

    summary["test_used_for_threshold_selection"] = False

    summary["model"] = str(MODEL_PATH)

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Print final report
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Test samples:                    {summary['test_samples']}")
    print(f"Labels:                          {summary['num_labels']}")
    print(
        f"Labels with test positives:      "
        f"{summary['labels_with_test_positive_support']}"
    )
    print(
        f"Labels with zero test positives: "
        f"{summary['labels_with_test_zero_support']}"
    )

    print("\nClassification metrics:")
    print(
        f"Macro Precision: {summary['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall:    {summary['macro_recall']:.4f}"
    )
    print(
        f"Macro F1:        {summary['macro_f1']:.4f}"
    )
    print(
        f"Micro Precision: {summary['micro_precision']:.4f}"
    )
    print(
        f"Micro Recall:    {summary['micro_recall']:.4f}"
    )
    print(
        f"Micro F1:        {summary['micro_f1']:.4f}"
    )

    print("\nRanking metrics:")
    print(
        f"Mean per-label ROC-AUC: "
        f"{summary['mean_per_label_roc_auc']:.4f}"
        if summary["mean_per_label_roc_auc"] is not None
        else "Mean per-label ROC-AUC: N/A"
    )
    print(
        f"Mean per-label PR-AUC:  "
        f"{summary['mean_per_label_pr_auc']:.4f}"
        if summary["mean_per_label_pr_auc"] is not None
        else "Mean per-label PR-AUC: N/A"
    )

    print("\nPer-label results:")
    print("-" * 70)

    display_columns = [
        "label",
        "support",
        "threshold",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    print(
        metrics_df[display_columns].to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(f"  {METRICS_CSV}")
    print(f"  {SUMMARY_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
