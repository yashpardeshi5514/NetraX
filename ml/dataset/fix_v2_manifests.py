import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\Projects\NetraX\ml\dataset\v2_manifests")

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
print("FIXING ODIR SUPERVISION MASKS")
print("=" * 70)

for split in ["train", "validation", "test"]:

    path = ROOT / f"{split}_manifest.csv"

    df = pd.read_csv(path)

    odir_mask = df["source"] == "ODIR"

    # For ODIR, only explicitly mapped positive labels are known.
    # Unmapped labels are ignored by the loss.
    for label in ODIR_MAPPED:

        df.loc[odir_mask, f"{label}_known"] = (
            df.loc[odir_mask, label] == 1
        ).astype(int)

    # ODIR labels outside the supported mapping remain unknown.
    for label in LABELS:
        if label not in ODIR_MAPPED:
            df.loc[odir_mask, label] = 0
            df.loc[odir_mask, f"{label}_known"] = 0

    output = ROOT / f"{split}_manifest_fixed.csv"

    df.to_csv(output, index=False)

    print()
    print(split.upper())
    print("Total:", len(df))

    print()
    print("ODIR known positives:")

    for label in ODIR_MAPPED:
        known = int(
            df.loc[
                (df["source"] == "ODIR") &
                (df[f"{label}_known"] == 1)
            ].shape[0]
        )

        positives = int(
            df.loc[
                (df["source"] == "ODIR") &
                (df[label] == 1)
            ].shape[0]
        )

        print(
            "{:<6} known={:<5} positive={:<5}".format(
                label,
                known,
                positives
            )
        )

    print("Saved:", output)

print()
print("=" * 70)
print("FIXED MANIFESTS CREATED")
print("=" * 70)
