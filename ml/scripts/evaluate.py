import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import (
    LABEL_COLUMNS,
    NUM_CLASSES,
    BATCH_SIZE,
    create_dataset,
)
from model import build_model


# ============================================================
# NetraX — Evaluation & Threshold Analysis
# ============================================================

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "netrax_best.keras",
    )
)

THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.05,
    dtype=np.float32,
)


def collect_predictions(model, dataset):
    """Collect labels and probabilities from a dataset."""
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


def safe_roc_auc(y_true, y_prob):
    """
    Calculate per-label ROC-AUC.

    Returns NaN when a label has only one class in the evaluation
    split, because ROC-AUC is undefined in that situation.
    """
    scores = []

    for index in range(NUM_CLASSES):
        if len(np.unique(y_true[:, index])) < 2:
            scores.append(np.nan)
        else:
            scores.append(
                roc_auc_score(
                    y_true[:, index],
                    y_prob[:, index],
                )
            )

    return np.asarray(scores, dtype=np.float64)


def safe_average_precision(y_true, y_prob):
    """
    Calculate per-label PR-AUC / Average Precision.

    Returns NaN when the label has no positive examples.
    """
    scores = []

    for index in range(NUM_CLASSES):
        if np.sum(y_true[:, index]) == 0:
            scores.append(np.nan)
        else:
            scores.append(
                average_precision_score(
                    y_true[:, index],
                    y_prob[:, index],
                )
            )

    return np.asarray(scores, dtype=np.float64)


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate macro/micro and per-label classification metrics."""
    y_pred = (y_prob >= threshold).astype(np.int32)

    precision = precision_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    return {
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "micro_precision": precision_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "micro_recall": recall_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "precision_per_label": precision,
        "recall_per_label": recall,
        "f1_per_label": f1,
    }


def find_best_threshold(y_true, y_prob):
    """
    Find a global probability threshold using validation data.

    The threshold is selected using macro F1. It must be selected
    before evaluating the untouched test set.
    """
    results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            y_true,
            y_prob,
            threshold=float(threshold),
        )

        results.append(
            {
                "threshold": float(threshold),
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
            }
        )

    table = pd.DataFrame(results)

    best_row = table.loc[
        table["macro_f1"].idxmax()
    ]

    return float(best_row["threshold"]), table


def print_report(
    y_true,
    y_prob,
    threshold,
    split_name,
):
    """Print a readable evaluation report."""
    metrics = calculate_metrics(
        y_true,
        y_prob,
        threshold=threshold,
    )

    roc_auc = safe_roc_auc(
        y_true,
        y_prob,
    )

    pr_auc = safe_average_precision(
        y_true,
        y_prob,
    )

    print("\n" + "=" * 70)
    print(
        f"NETRAX — {split_name.upper()} EVALUATION"
    )
    print("=" * 70)

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print("\nAggregate metrics")
    print("-" * 70)
    print(
        f"Macro Precision : "
        f"{metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall    : "
        f"{metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1        : "
        f"{metrics['macro_f1']:.4f}"
    )
    print(
        f"Micro Precision : "
        f"{metrics['micro_precision']:.4f}"
    )
    print(
        f"Micro Recall    : "
        f"{metrics['micro_recall']:.4f}"
    )
    print(
        f"Micro F1        : "
        f"{metrics['micro_f1']:.4f}"
    )

    print("\nPer-label metrics")
    print("-" * 70)

    rows = []

    for index, label in enumerate(LABEL_COLUMNS):
        support = int(
            np.sum(y_true[:, index])
        )

        rows.append(
            {
                "label": label,
                "support": support,
                "precision": metrics[
                    "precision_per_label"
                ][index],
                "recall": metrics[
                    "recall_per_label"
                ][index],
                "f1": metrics[
                    "f1_per_label"
                ][index],
                "roc_auc": roc_auc[index],
                "pr_auc": pr_auc[index],
            }
        )

    report = pd.DataFrame(rows)

    print(
        report.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return report


def dry_run():
    """
    Verify the evaluation pipeline without requiring a trained model.

    A fresh model is used only to verify shapes and metric functions.
    """
    print("=" * 70)
    print("NETRAX — EVALUATION PIPELINE DRY RUN")
    print("=" * 70)

    model = build_model(
        image_size=384,
        num_classes=NUM_CLASSES,
    )

    validation_dataset = create_dataset(
        "validation",
        batch_size=BATCH_SIZE,
        training=False,
    )

    images, labels = next(
        iter(validation_dataset)
    )

    probabilities = model(
        images,
        training=False,
    ).numpy()

    y_true = labels.numpy()

    print("\nInput shape:", images.shape)
    print("True-label shape:", y_true.shape)
    print("Probability shape:", probabilities.shape)

    assert y_true.shape[-1] == NUM_CLASSES
    assert probabilities.shape[-1] == NUM_CLASSES

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    print(
        "\nBatch macro F1:",
        f"{metrics['macro_f1']:.4f}",
    )

    print(
        "Batch micro F1:",
        f"{metrics['micro_f1']:.4f}",
    )

    roc_auc = safe_roc_auc(
        y_true,
        probabilities,
    )

    pr_auc = safe_average_precision(
        y_true,
        probabilities,
    )

    print(
        "Defined ROC-AUC labels:",
        int(np.sum(~np.isnan(roc_auc))),
        "/",
        NUM_CLASSES,
    )

    print(
        "Defined PR-AUC labels:",
        int(np.sum(~np.isnan(pr_auc))),
        "/",
        NUM_CLASSES,
    )

    print("\nThreshold search test...")

    best_threshold, threshold_table = find_best_threshold(
        y_true,
        probabilities,
    )

    print(
        "Best batch threshold:",
        f"{best_threshold:.2f}",
    )

    print(
        "Threshold candidates:",
        len(threshold_table),
    )

    print("\n" + "=" * 70)
    print("EVALUATION PIPELINE DRY RUN PASSED")
    print("=" * 70)

    print(
        "\nNote: this script does not evaluate the real model yet."
    )
    print(
        "The actual test set must remain untouched until training "
        "and validation threshold selection are complete."
    )


def evaluate_trained_model():
    """
    Evaluate an already-trained model.

    This function is intentionally separate from dry_run().
    It uses validation data for threshold selection and the test
    data only for the final evaluation.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}\n\n"
            "Run training before calling evaluate_trained_model()."
        )

    print(
        f"Loading model:\n{MODEL_PATH}"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    validation_dataset = create_dataset(
        "validation",
        batch_size=BATCH_SIZE,
        training=False,
    )

    test_dataset = create_dataset(
        "test",
        batch_size=BATCH_SIZE,
        training=False,
    )

    print("\nCollecting validation predictions...")
    val_true, val_prob = collect_predictions(
        model,
        validation_dataset,
    )

    best_threshold, threshold_table = find_best_threshold(
        val_true,
        val_prob,
    )

    print(
        f"\nSelected validation threshold: "
        f"{best_threshold:.2f}"
    )

    print("\nCollecting test predictions...")
    test_true, test_prob = collect_predictions(
        model,
        test_dataset,
    )

    report = print_report(
        test_true,
        test_prob,
        best_threshold,
        "test",
    )

    output_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "evaluation",
            "test_metrics.csv",
        )
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    report.to_csv(
        output_path,
        index=False,
    )

    threshold_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "evaluation",
            "threshold_search.csv",
        )
    )

    threshold_table.to_csv(
        threshold_path,
        index=False,
    )

    print(
        f"\nSaved per-label metrics:\n{output_path}"
    )

    print(
        f"Saved threshold search:\n{threshold_path}"
    )


if __name__ == "__main__":
    evaluate_trained_model()
