import numpy as np
import tensorflow as tf

NUM_CLASSES = 45

LABELS = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS",
    "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
    "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP",
    "CWS", "CB", "ODPM", "PRH", "MNF", "HR", "CRAO", "TD", "CME",
    "PTCR", "CF", "VH", "MCA", "VS", "BRAO", "PLQ", "HPED", "CL"
]


def calculate_masked_class_weights(
    targets,
    masks,
    max_weight=10.0
):
    """
    Calculate positive-class weights using only known labels.

    targets:
        numpy array [N, 45]

    masks:
        numpy array [N, 45]

    Unknown labels are excluded from the statistics.
    """

    targets = np.asarray(targets, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)

    if targets.shape != masks.shape:
        raise ValueError(
            f"targets shape {targets.shape} != masks shape {masks.shape}"
        )

    weights = np.ones(NUM_CLASSES, dtype=np.float32)

    for i in range(NUM_CLASSES):

        known = masks[:, i] > 0.5

        total_known = int(known.sum())

        if total_known == 0:
            weights[i] = 1.0
            continue

        positive = int(
            targets[known, i].sum()
        )

        negative = total_known - positive

        if positive == 0:
            weights[i] = 1.0
        else:
            weight = negative / positive
            weights[i] = min(
                max(weight, 1.0),
                max_weight
            )

    return weights


def masked_weighted_binary_crossentropy(
    y_true,
    y_pred,
    mask,
    class_weights
):
    """
    Masked weighted BCE.

    y_true:
        [batch, 45]

    y_pred:
        [batch, 45]

    mask:
        [batch, 45]

    class_weights:
        [45]

    Unknown labels contribute zero loss.
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    class_weights = tf.cast(
        class_weights,
        tf.float32
    )

    epsilon = tf.keras.backend.epsilon()

    y_pred = tf.clip_by_value(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    positive_weight = class_weights

    loss = -(
        positive_weight * y_true *
        tf.math.log(y_pred)
        +
        (1.0 - y_true) *
        tf.math.log(1.0 - y_pred)
    )

    loss = loss * mask

    known_count = tf.reduce_sum(mask)

    loss = tf.reduce_sum(loss) / tf.maximum(
        known_count,
        1.0
    )

    return loss


def masked_macro_binary_accuracy(
    y_true,
    y_pred,
    mask,
    threshold=0.5
):
    """
    Macro binary accuracy calculated only
    over known labels.
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred >= threshold, tf.float32)
    mask = tf.cast(mask, tf.float32)

    correct = tf.cast(
        tf.equal(y_true, y_pred),
        tf.float32
    )

    correct = correct * mask

    known = tf.reduce_sum(mask)

    return tf.reduce_sum(correct) / tf.maximum(
        known,
        1.0
    )


if __name__ == "__main__":

    print("=" * 70)
    print("V2 MASKED LOSS TEST")
    print("=" * 70)

    # Small synthetic example.
    #
    # Sample 0:
    #   all labels known
    #
    # Sample 1:
    #   only first two labels known
    #
    # Sample 2:
    #   only label 0 known

    y_true = tf.constant([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=tf.float32)

    y_pred = tf.constant([
        [0.9, 0.1, 0.8, 0.2],
        [0.8, 0.7, 0.4, 0.3],
        [0.2, 0.8, 0.3, 0.4],
    ], dtype=tf.float32)

    mask = tf.constant([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=tf.float32)

    weights = tf.ones(4, dtype=tf.float32)

    loss = masked_weighted_binary_crossentropy(
        y_true,
        y_pred,
        mask,
        weights
    )

    accuracy = masked_macro_binary_accuracy(
        y_true,
        y_pred,
        mask
    )

    print()
    print("Loss:", float(loss.numpy()))
    print("Masked accuracy:", float(accuracy.numpy()))

    print()
    print("Testing unknown-label behavior...")

    # Change only unknown predictions.
    y_pred_changed = tf.constant([
        [0.9, 0.1, 0.8, 0.2],
        [0.8, 0.7, 0.99, 0.99],
        [0.2, 0.8, 0.99, 0.99],
    ], dtype=tf.float32)

    loss_changed = masked_weighted_binary_crossentropy(
        y_true,
        y_pred_changed,
        mask,
        weights
    )

    print(
        "Original loss:",
        float(loss.numpy())
    )

    print(
        "Changed-unknown loss:",
        float(loss_changed.numpy())
    )

    difference = abs(
        float(loss.numpy()) -
        float(loss_changed.numpy())
    )

    print(
        "Difference:",
        difference
    )

    if difference < 1e-6:
        print("PASS: unknown labels are ignored.")
    else:
        raise RuntimeError(
            "FAIL: unknown labels affect the loss."
        )

    print()
    print("=" * 70)
    print("MASKED LOSS TEST PASSED")
    print("=" * 70)
