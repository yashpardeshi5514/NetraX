import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "ml" / "scripts")
)

from v2_dataset import create_dataset


MODEL_PATH = (
    ROOT
    / "ml"
    / "models"
    / "netrax_v2_best.keras"
)

OUTPUT_DIR = (
    ROOT
    / "ml"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RFMiD LABELS
# ============================================================

LABELS = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS",
    "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
    "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP",
    "CWS", "CB", "ODPM", "PRH", "MNF", "HR", "CRAO", "TD", "CME",
    "PTCR", "CF", "VH", "MCA", "VS", "BRAO", "PLQ", "HPED", "CL"
]


# ============================================================
# GET PREDICTIONS
# ============================================================

def get_predictions(
    model,
    split
):

    print(
        f"Loading RFMiD {split} dataset..."
    )

    # IMPORTANT:
    #
    # v2_dataset.create_dataset()
    # returns:
    #
    #     dataset, dataframe
    #
    dataset, df = create_dataset(
        split=split,
        source="RFMiD",
        training=False,
        shuffle=False
    )

    all_targets = []
    all_predictions = []

    batch_count = 0

    # The dataset itself returns:
    #
    #     images,
    #     {
    #         "targets": target,
    #         "mask": mask
    #     }

    for images, labels in dataset:

        batch_count += 1

        if not isinstance(
            labels,
            dict
        ):

            raise TypeError(
                "Expected dataset labels "
                "to be a dictionary."
            )

        if "targets" not in labels:

            raise KeyError(
                "Dataset output does not contain "
                "'targets'."
            )

        targets = labels[
            "targets"
        ]

        predictions = model(
            images,
            training=False
        )

        all_targets.append(
            targets.numpy()
        )

        all_predictions.append(
            predictions.numpy()
        )

    if batch_count == 0:

        raise RuntimeError(
            f"No batches were produced "
            f"for RFMiD {split}."
        )

    y_true = np.concatenate(
        all_targets,
        axis=0
    ).astype(
        np.int32
    )

    y_prob = np.concatenate(
        all_predictions,
        axis=0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if y_true.shape != y_prob.shape:

        raise ValueError(
            "Target and prediction shapes "
            "do not match:\n"
            f"Targets: {y_true.shape}\n"
            f"Predictions: {y_prob.shape}"
        )

    if y_true.shape[1] != len(LABELS):

        raise ValueError(
            "Unexpected number of labels:\n"
            f"Found: {y_true.shape[1]}\n"
            f"Expected: {len(LABELS)}"
        )

    print(
        f"{split}: "
        f"{len(df)} images, "
        f"{y_true.shape[1]} labels, "
        f"{batch_count} batches"
    )

    return (
        y_true,
        y_prob
    )


# ============================================================
# FIND VALIDATION THRESHOLDS
# ============================================================

def find_thresholds(
    y_true,
    y_prob
):

    print()
    print("=" * 70)
    print("FINDING VALIDATION THRESHOLDS")
    print("=" * 70)

    thresholds = np.full(
        len(LABELS),
        0.50,
        dtype=np.float32
    )

    threshold_grid = np.arange(
        0.05,
        0.951,
        0.01
    )

    for i, label in enumerate(
        LABELS
    ):

        positives = int(
            np.sum(
                y_true[:, i]
            )
        )

        negatives = int(
            len(y_true[:, i])
            - positives
        )

        # ----------------------------------------------------
        # No positive validation examples
        # ----------------------------------------------------

        if positives == 0:

            thresholds[i] = 0.50

            print(
                f"{label:>5} | "
                f"support={positives:>3} | "
                f"threshold=0.50 | "
                f"no positive validation samples"
            )

            continue

        best_threshold = 0.50
        best_f1 = -1.0

        # ----------------------------------------------------
        # Search threshold using validation ONLY
        # ----------------------------------------------------

        for threshold in threshold_grid:

            predictions = (
                y_prob[:, i]
                >= threshold
            ).astype(
                np.int32
            )

            score = f1_score(
                y_true[:, i],
                predictions,
                zero_division=0
            )

            if score > best_f1:

                best_f1 = score
                best_threshold = threshold

        thresholds[i] = (
            best_threshold
        )

        print(
            f"{label:>5} | "
            f"support={positives:>3} | "
            f"negative={negatives:>3} | "
            f"threshold={best_threshold:.2f} | "
            f"val_F1={best_f1:.4f}"
        )

    return thresholds


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(
    y_true,
    y_prob,
    thresholds
):

    # --------------------------------------------------------
    # Apply thresholds derived from validation
    # --------------------------------------------------------

    y_pred = (
        y_prob
        >= thresholds[None, :]
    ).astype(
        np.int32
    )

    # --------------------------------------------------------
    # Macro metrics
    # --------------------------------------------------------

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Micro metrics
    # --------------------------------------------------------

    (
        micro_precision,
        micro_recall,
        micro_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Per-label metrics
    # --------------------------------------------------------

    rows = []

    for i, label in enumerate(
        LABELS
    ):

        true_values = y_true[:, i]

        probabilities = y_prob[:, i]

        predictions = y_pred[:, i]

        (
            precision,
            recall,
            f1,
            _
        ) = precision_recall_fscore_support(
            true_values,
            predictions,
            average="binary",
            zero_division=0
        )

        support = int(
            np.sum(
                true_values
            )
        )

        negative_support = int(
            len(true_values)
            - support
        )

        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        if (
            support > 0
            and negative_support > 0
        ):

            roc_auc = roc_auc_score(
                true_values,
                probabilities
            )

        else:

            roc_auc = np.nan

        # ----------------------------------------------------
        # PR-AUC
        # ----------------------------------------------------

        if support > 0:

            pr_auc = (
                average_precision_score(
                    true_values,
                    probabilities
                )
            )

        else:

            pr_auc = np.nan

        rows.append(
            {
                "label": label,
                "support": support,
                "threshold": float(
                    thresholds[i]
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1": float(
                    f1
                ),
                "roc_auc": (
                    float(roc_auc)
                    if not np.isnan(
                        roc_auc
                    )
                    else np.nan
                ),
                "pr_auc": (
                    float(pr_auc)
                    if not np.isnan(
                        pr_auc
                    )
                    else np.nan
                ),
            }
        )

    results_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Mean ROC-AUC / PR-AUC
    # --------------------------------------------------------

    mean_roc_auc = (
        results_df[
            "roc_auc"
        ]
        .dropna()
        .mean()
    )

    mean_pr_auc = (
        results_df[
            "pr_auc"
        ]
        .dropna()
        .mean()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "test_images": int(
            len(y_true)
        ),
        "number_of_labels": int(
            len(LABELS)
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "micro_precision": float(
            micro_precision
        ),
        "micro_recall": float(
            micro_recall
        ),
        "micro_f1": float(
            micro_f1
        ),
        "mean_roc_auc": float(
            mean_roc_auc
        ),
        "mean_pr_auc": float(
            mean_pr_auc
        ),
    }

    return (
        summary,
        results_df
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "NetraX V2 — RFMiD TEST EVALUATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    print()
    print("Model:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print(
        "Loading V2 best model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STEP 1 — RFMiD VALIDATION"
    )
    print("=" * 70)

    y_val, p_val = get_predictions(
        model,
        "validation"
    )

    # --------------------------------------------------------
    # Validation-derived thresholds
    # --------------------------------------------------------

    thresholds = find_thresholds(
        y_val,
        p_val
    )

    threshold_path = (
        OUTPUT_DIR
        / "v2_rfmid_validation_thresholds.json"
    )

    threshold_dict = {
        label: float(threshold)
        for label, threshold in zip(
            LABELS,
            thresholds
        )
    }

    with open(
        threshold_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_dict,
            file,
            indent=2
        )

    print()
    print(
        "Validation thresholds saved:"
    )

    print(
        threshold_path
    )

    # ========================================================
    # UNTOUCHED TEST
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STEP 2 — UNTOUCHED RFMiD TEST"
    )
    print("=" * 70)

    y_test, p_test = get_predictions(
        model,
        "test"
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    summary, results_df = (
        evaluate_test(
            y_test,
            p_test,
            thresholds
        )
    )

    # --------------------------------------------------------
    # Save per-label metrics
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "v2_rfmid_test_metrics.csv"
    )

    results_df.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "v2_rfmid_test_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # ========================================================
    # V2 TEST RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "V2 TEST RESULTS"
    )
    print("=" * 70)

    print(
        f"Test Images       : "
        f"{summary['test_images']}"
    )

    print(
        f"Labels            : "
        f"{summary['number_of_labels']}"
    )

    print()

    print(
        f"Macro Precision   : "
        f"{summary['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{summary['macro_recall']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{summary['macro_f1']:.4f}"
    )

    print()

    print(
        f"Micro Precision   : "
        f"{summary['micro_precision']:.4f}"
    )

    print(
        f"Micro Recall      : "
        f"{summary['micro_recall']:.4f}"
    )

    print(
        f"Micro F1          : "
        f"{summary['micro_f1']:.4f}"
    )

    print()

    print(
        f"Mean ROC-AUC      : "
        f"{summary['mean_roc_auc']:.4f}"
    )

    print(
        f"Mean PR-AUC       : "
        f"{summary['mean_pr_auc']:.4f}"
    )

    # ========================================================
    # V1 REFERENCE
    # ========================================================

    v1 = {
        "macro_precision": 0.1526,
        "macro_recall": 0.2115,
        "macro_f1": 0.1687,
        "micro_precision": 0.3789,
        "micro_recall": 0.5964,
        "micro_f1": 0.4634,
        "mean_roc_auc": 0.8111,
        "mean_pr_auc": 0.2435,
    }

    # --------------------------------------------------------
    # V1 vs V2
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "V1 REFERENCE VS V2"
    )
    print("=" * 70)

    comparison_rows = []

    for metric in v1:

        v1_value = v1[
            metric
        ]

        v2_value = summary[
            metric
        ]

        change = (
            v2_value
            - v1_value
        )

        comparison_rows.append(
            {
                "metric": metric,
                "V1": v1_value,
                "V2": v2_value,
                "change": change,
            }
        )

        print(
            f"{metric:<18} "
            f"V1={v1_value:.4f}  "
            f"V2={v2_value:.4f}  "
            f"Change={change:+.4f}"
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        OUTPUT_DIR
        / "v1_vs_v2_rfmid_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # ========================================================
    # PER-LABEL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "PER-LABEL TEST RESULTS"
    )
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.4f}"
            )
        )
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()
    print("=" * 70)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 70)

    print()
    print("Created files:")

    print(
        f"1. {threshold_path}"
    )

    print(
        f"2. {metrics_path}"
    )

    print(
        f"3. {summary_path}"
    )

    print(
        f"4. {comparison_path}"
    )


if __name__ == "__main__":

    main()