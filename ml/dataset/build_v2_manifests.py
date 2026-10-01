import pandas as pd
from pathlib import Path

RFMiD_ROOT = Path(r"D:\Datasets\RFMiD")
ODIR_ROOT = Path(r"D:\Projects\NetraX\ml\dataset\odir_v2")
OUTPUT = Path(r"D:\Projects\NetraX\ml\dataset\v2_manifests")
OUTPUT.mkdir(parents=True, exist_ok=True)

LABELS = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS",
    "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
    "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP",
    "CWS", "CB", "ODPM", "PRH", "MNF", "HR", "CRAO", "TD", "CME",
    "PTCR", "CF", "VH", "MCA", "VS", "BRAO", "PLQ", "HPED", "CL"
]

ODIR_MAPPED = [
    "DR", "ARMD", "MYA", "ERM", "DN",
    "BRVO", "MNF", "TSLN", "LS"
]

print("=" * 70)
print("NETRAX V2 COMBINED MANIFEST")
print("=" * 70)

# ---------------------------------------------------------
# RFMiD
# ---------------------------------------------------------

rfmid_splits = {
    "train": RFMiD_ROOT / "Training_Set" / "RFMiD_Training_Labels.csv",
    "validation": RFMiD_ROOT / "Evaluation_Set" / "RFMiD_Validation_Labels.csv",
    "test": RFMiD_ROOT / "Test_Set" / "RFMiD_Testing_Labels.csv",
}

rfmid_image_dirs = {
    "train": RFMiD_ROOT / "Training_Set" / "Training",
    "validation": RFMiD_ROOT / "Evaluation_Set" / "Validation",
    "test": RFMiD_ROOT / "Test_Set" / "Test",
}

combined = {}

for split in ["train", "validation", "test"]:

    df = pd.read_csv(rfmid_splits[split])

    records = []

    for _, row in df.iterrows():

        image_id = str(row["ID"])

        # RFMiD files are PNG files.
        filename = image_id + ".png"

        image_path = rfmid_image_dirs[split] / filename

        if not image_path.exists():
            raise FileNotFoundError(
                f"RFMiD image missing: {image_path}"
            )

        record = {
            "source": "RFMiD",
            "split": split,
            "patient_id": "",
            "filename": filename,
            "relative_path": str(
                image_path.relative_to(RFMiD_ROOT)
            ),
        }

        # RFMiD provides authoritative labels for all 45 classes.
        for label in LABELS:
            record[label] = int(row[label])

        # Every RFMiD label is authoritative.
        for label in LABELS:
            record[f"{label}_known"] = 1

        records.append(record)

    combined[split] = pd.DataFrame(records)

    print()
    print("RFMiD", split, ":", len(records))

# ---------------------------------------------------------
# ODIR
# ---------------------------------------------------------

for split in ["train", "validation", "test"]:

    csv_path = ODIR_ROOT / f"{split}_labels.csv"

    odir = pd.read_csv(csv_path)

    records = []

    for _, row in odir.iterrows():

        record = {
            "source": "ODIR",
            "split": split,
            "patient_id": int(row["patient_id"]),
            "filename": row["filename"],
            "relative_path": str(
                Path("odir_v2") / split / row["filename"]
            ),
        }

        mapped = set(
            str(row["mapped_labels"]).split(",")
        ) if pd.notna(row["mapped_labels"]) else set()

        for label in LABELS:

            if label in ODIR_MAPPED:
                # For mapped ODIR labels, the mapping is authoritative
                # only for labels explicitly detected.
                record[label] = int(label in mapped)
                record[f"{label}_known"] = 1
            else:
                # No evidence either way.
                # Do NOT turn this into a negative label.
                record[label] = 0
                record[f"{label}_known"] = 0

        records.append(record)

    odir_df = pd.DataFrame(records)

    combined[split] = pd.concat(
        [combined[split], odir_df],
        ignore_index=True
    )

    print("ODIR ", split, ":", len(records))

# ---------------------------------------------------------
# Save manifests
# ---------------------------------------------------------

for split, df in combined.items():

    output = OUTPUT / f"{split}_manifest.csv"

    df.to_csv(output, index=False)

    print()
    print(split.upper())
    print("Total records:", len(df))
    print("RFMiD:", int((df["source"] == "RFMiD").sum()))
    print("ODIR:", int((df["source"] == "ODIR").sum()))

    print()
    print("Known-label counts:")

    for label in LABELS:
        known = int(df[f"{label}_known"].sum())
        positives = int(
            df.loc[df[f"{label}_known"] == 1, label].sum()
        )

        if known > 0:
            print(
                "{:<6} known={:<5} positive={:<5}".format(
                    label,
                    known,
                    positives
                )
            )

print()
print("=" * 70)
print("MANIFESTS CREATED")
print("=" * 70)
print(OUTPUT)
