from pathlib import Path
import pandas as pd


DATASET_ROOT = Path(r"D:\Datasets\RFMiD")

SPLITS = {
    "train": DATASET_ROOT / "Training_Set" / "RFMiD_Training_Labels.csv",
    "validation": DATASET_ROOT / "Evaluation_Set" / "RFMiD_Validation_Labels.csv",
    "test": DATASET_ROOT / "Test_Set" / "RFMiD_Testing_Labels.csv",
}


DISEASE_COLUMNS = None

results = []


for split_name, csv_path in SPLITS.items():

    df = pd.read_csv(csv_path)

    if DISEASE_COLUMNS is None:
        DISEASE_COLUMNS = [
            column
            for column in df.columns
            if column not in ["ID", "Disease_Risk"]
        ]

    for disease in DISEASE_COLUMNS:

        positives = int(df[disease].sum())
        negatives = len(df) - positives

        results.append({
            "split": split_name,
            "label": disease,
            "total": len(df),
            "positive": positives,
            "negative": negatives,
            "positive_percent": round(
                positives / len(df) * 100,
                2
            ),
        })


result_df = pd.DataFrame(results)


print("\n" + "=" * 80)
print("NETRAX — LABEL SUPPORT ANALYSIS")
print("=" * 80)


for split_name in SPLITS:

    print(f"\n{'-' * 80}")
    print(split_name.upper())
    print("-" * 80)

    split_df = result_df[
        result_df["split"] == split_name
    ].sort_values(
        "positive",
        ascending=False
    )

    print(
        split_df[
            [
                "label",
                "positive",
                "negative",
                "positive_percent"
            ]
        ].to_string(index=False)
    )


print("\n" + "=" * 80)
print("GLOBAL LABEL SUPPORT")
print("=" * 80)


global_df = (
    result_df
    .groupby("label")
    .agg(
        positive=("positive", "sum"),
        negative=("negative", "sum")
    )
    .reset_index()
)

global_df["total"] = (
    global_df["positive"] +
    global_df["negative"]
)

global_df["positive_percent"] = (
    global_df["positive"] /
    global_df["total"] *
    100
)

global_df = global_df.sort_values(
    "positive",
    ascending=False
)


print(
    global_df[
        [
            "label",
            "positive",
            "negative",
            "positive_percent"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 80)
print("SUPPORT GROUPS")
print("=" * 80)


groups = {
    ">= 100 positives": global_df[
        global_df["positive"] >= 100
    ],

    "50-99 positives": global_df[
        (global_df["positive"] >= 50) &
        (global_df["positive"] < 100)
    ],

    "20-49 positives": global_df[
        (global_df["positive"] >= 20) &
        (global_df["positive"] < 50)
    ],

    "10-19 positives": global_df[
        (global_df["positive"] >= 10) &
        (global_df["positive"] < 20)
    ],

    "< 10 positives": global_df[
        global_df["positive"] < 10
    ],
}


for name, group in groups.items():

    print(f"\n{name}: {len(group)} labels")

    if len(group) > 0:
        print(
            ", ".join(group["label"].tolist())
        )


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)