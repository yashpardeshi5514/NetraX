import tensorflow as tf


def create_class_weights(positive_counts, total_samples, max_weight=10.0):
    """
    Create capped positive-class weights for multi-label classification.

    Formula:
        weight = negatives / positives

    The resulting weight is capped to avoid instability caused by
    extremely rare RFMiD labels.
    """

    positive_counts = tf.cast(positive_counts, tf.float32)
    total_samples = tf.cast(total_samples, tf.float32)

    negative_counts = total_samples - positive_counts

    weights = tf.math.divide_no_nan(
        negative_counts,
        positive_counts
    )

    weights = tf.clip_by_value(
        weights,
        clip_value_min=1.0,
        clip_value_max=max_weight
    )

    return weights


def weighted_binary_crossentropy(class_weights):
    """
    Weighted binary cross-entropy for 45-label multi-label classification.
    """

    class_weights = tf.cast(class_weights, tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )

        bce = -(
            y_true * tf.math.log(y_pred) +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        weights = (
            y_true * class_weights +
            (1.0 - y_true)
        )

        weighted_loss = bce * weights

        return tf.reduce_mean(weighted_loss)

    return loss