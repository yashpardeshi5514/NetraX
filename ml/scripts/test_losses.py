import numpy as np
import tensorflow as tf

from losses import create_class_weights, weighted_binary_crossentropy


# RFMiD has exactly 45 disease labels.
# Counts are in the same order as LABEL_COLUMNS in dataset.py.
positive_counts = np.array([
    632, 169, 523, 230, 167,
    119, 304, 26, 79, 27,
    61, 445, 45, 10, 25,
    115, 96, 11, 26, 17,
    25, 71, 54, 24, 32,
    17, 10, 8, 2, 2,
    2, 5, 3, 1, 4,
    4, 9, 7, 6, 4,
    1, 4, 4, 2, 1
], dtype=np.int32)

TOTAL_SAMPLES = 1920

print("=" * 70)
print("NETRAX — LOSS FUNCTION TEST")
print("=" * 70)

print("\nNumber of classes:", len(positive_counts))

assert len(positive_counts) == 45, (
    f"Expected 45 classes, got {len(positive_counts)}"
)

weights = create_class_weights(
    positive_counts,
    TOTAL_SAMPLES,
    max_weight=10.0
)

print("\nClass weights:")
for i, weight in enumerate(weights.numpy()):
    print(f"{i:02d}: {weight:.2f}")

print("\nWeight range:")
print("Minimum:", float(tf.reduce_min(weights)))
print("Maximum:", float(tf.reduce_max(weights)))

# Test loss with fake 45-label predictions.
y_true = tf.constant(
    [[1, 0, 1] + [0] * 42],
    dtype=tf.float32
)

y_pred = tf.constant(
    [[0.8, 0.2, 0.7] + [0.1] * 42],
    dtype=tf.float32
)

assert y_true.shape[-1] == 45
assert y_pred.shape[-1] == 45
assert weights.shape[-1] == 45

loss_fn = weighted_binary_crossentropy(weights)

loss = loss_fn(y_true, y_pred)

print("\nTest loss:", float(loss))

print("\nLoss test completed successfully.")
