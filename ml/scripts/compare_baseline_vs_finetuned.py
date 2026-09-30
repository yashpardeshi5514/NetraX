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
# NetraX — Baseline vs Fine-Tuned Test Evaluation
# ============================================================

IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_CLASSES = 45

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "evaluation"

BASELINE_MODEL = MODEL_DIR / "netrax_best.keras"
FINETUNED_MODEL = MODEL_DIR / "netrax_finetuned_best.keras"

RESULTS_CSV = EVAL_DIR / "model_comparison_test_metrics.csv"
SUMMARY_JSON = EVAL_DIR / "model_comparison_summary.json"


def collect_labels(dataset):
    labels = []

    for _, y in dataset:
        labels.append(y.numpy())

    return np.concatenate(labels, axis=0).astype(np.float32)


def collect_predictions(model, dataset):
    predictions = model.predict(dataset, verbose=1)
    return np.asarray(predictions, dtype=np.float32)


def find_best_global_threshold(y_true, probabilities):
    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.50
    best_macro_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int32)

        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        if f1 > best_macro_f1:
            best_macro_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, best_macro_f1


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int32)

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

    per_precision, per_recall, per_f1, per_support = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average=None,
            zero_division=0,
        )
    )

    rows = []

    for index, label in enumerate(LABEL_COLUMNS):
        true_label = y_true[:, index]
        probability_label = probabilities[:, index]

        support = int(true_label.sum())

        if np.unique(true_label).size < 2:
            roc_auc = np.nan
            pr_auc = np.nan
        else:
            roc_auc = roc_auc_score(
                true_label,
                probability_label,
            )
            pr_auc = average_precision_score(
                true_label,
                probability_label,
            )

        rows.append(
            {
                "label": label,
                "support": support,
                "threshold": threshold,
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "roc_auc": float(roc_auc)
                if not np.isnan(roc_auc)
                else None,
                "pr_auc": float(pr_auc)
                if not np.isnan(pr_auc)
                else None,
            }
        )

    summary = {
        "threshold": float(threshold),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
    }

    return summary, rows


def evaluate_model(name, model_path, val_ds, test_ds, y_val, y_test):
    print("\n" + "=" * 70)
    print(f"Evaluating: {name}")
    print(f"Model: {model_path}")
    print("=" * 70)

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    print("\nValidation predictions...")
    val_probabilities = collect_predictions(model, val_ds)

    print("\nTest predictions...")
    test_probabilities = collect_predictions(model, test_ds)

    if val_probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"{name}: expected {NUM_CLASSES} outputs, "
            f"got {val_probabilities.shape[1]}"
        )

    # Threshold is selected using validation only.
    threshold, validation_f1 = find_best_global_threshold(
        y_val,
        val_probabilities,
    )

    print(f"\nValidation-selected threshold: {threshold:.2f}")
    print(f"Validation macro F1:            {validation_f1:.4f}")

    test_summary, per_label = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\nTEST RESULTS")
    print("-" * 70)

    for key, value in test_summary.items():
        print(f"{key:20s}: {value:.4f}" if key != "threshold"
              else f"{key:20s}: {value:.2f}")

    for row in per_label:
        row["model"] = name

    test_summary["model"] = name

    return test_summary, per_label


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NetraX — Baseline vs Fine-Tuned Evaluation")
    print("=" * 70)

    print("\nLoading validation dataset...")
    val_ds = create_dataset(
        "validation",
        batch_size=BATCH_SIZE,
        training=False,
    )

    print("Loading test dataset...")
    test_ds = create_dataset(
        "test",
        batch_size=BATCH_SIZE,
        training=False,
    )

    print("\nCollecting ground-truth validation labels...")
    y_val = collect_labels(val_ds)

    print("Collecting ground-truth test labels...")
    y_test = collect_labels(test_ds)

    if y_val.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Validation labels have {y_val.shape[1]} classes; "
            f"expected {NUM_CLASSES}."
        )

    if y_test.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Test labels have {y_test.shape[1]} classes; "
            f"expected {NUM_CLASSES}."
        )

    all_summaries = []
    all_rows = []

    summary, rows = evaluate_model(
        "baseline",
        BASELINE_MODEL,
        val_ds,
        test_ds,
        y_val,
        y_test,
    )

    all_summaries.append(summary)
    all_rows.extend(rows)

    summary, rows = evaluate_model(
        "fine_tuned",
        FINETUNED_MODEL,
        val_ds,
        test_ds,
        y_val,
        y_test,
    )

    all_summaries.append(summary)
    all_rows.extend(rows)

    # ---------------------------------------------------------
    # Save per-label test metrics
    # ---------------------------------------------------------
    metrics_df = pd.DataFrame(all_rows)

    column_order = [
        "model",
        "label",
        "support",
        "threshold",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    metrics_df = metrics_df[column_order]
    metrics_df.to_csv(RESULTS_CSV, index=False)

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------
    with open(SUMMARY_JSON, "w", encoding="utf-8") as file:
        json.dump(
            {
                "models": all_summaries,
                "test_samples": int(len(y_test)),
                "num_classes": NUM_CLASSES,
                "threshold_selection": (
                    "Global threshold selected independently "
                    "on validation data for each model."
                ),
                "test_set_used_only_for_final_evaluation": True,
            },
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------
    comparison = pd.DataFrame(all_summaries)

    print("\n" + "=" * 70)
    print("FINAL TEST COMPARISON")
    print("=" * 70)

    print(
        comparison[
            [
                "model",
                "threshold",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "micro_precision",
                "micro_recall",
                "micro_f1",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {RESULTS_CSV}")
    print(f"  {SUMMARY_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
