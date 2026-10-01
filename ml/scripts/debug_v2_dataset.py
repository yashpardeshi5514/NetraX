import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from v2_dataset import create_dataset


print("=" * 70)
print("V2 DATASET STRUCTURE DEBUG")
print("=" * 70)

dataset = create_dataset(
    split="validation",
    source="RFMiD",
    training=False
)

batch = next(iter(dataset))

print()
print("TYPE OF BATCH:")
print(type(batch))

print()
print("BATCH:")
print(batch)

print()
print("LENGTH:")
try:
    print(len(batch))
except Exception as e:
    print("Cannot calculate len:", e)

print()
print("STRUCTURE:")

if isinstance(batch, dict):

    print("Batch is a DICT")
    print("Keys:", list(batch.keys()))

    for key, value in batch.items():
        print(
            f"  {key}: "
            f"type={type(value)}, "
            f"shape={getattr(value, 'shape', None)}"
        )

elif isinstance(batch, (tuple, list)):

    print(
        f"Batch is a {type(batch).__name__} "
        f"with {len(batch)} elements"
    )

    for i, value in enumerate(batch):

        print(
            f"  [{i}]: "
            f"type={type(value)}, "
            f"shape={getattr(value, 'shape', None)}"
        )

else:

    print(
        "Batch is another type:",
        type(batch)
    )

print()
print("=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)