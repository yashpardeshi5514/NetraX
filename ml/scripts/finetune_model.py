import os

import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# NetraX — Fine-Tuning Model
# ============================================================

IMAGE_SIZE = 384
NUM_CLASSES = 45

# Number of final EfficientNetB0 layers to fine-tune.
# Earlier layers remain frozen to preserve general visual features.
FINE_TUNE_LAYERS = 30


def build_finetune_model(
    image_size=IMAGE_SIZE,
    num_classes=NUM_CLASSES,
    pretrained_model_path=None,
):
    """
    Build a fine-tuning model from the existing NetraX baseline.

    The EfficientNetB0 backbone starts from the already-trained
    NetraX model. Its final FINE_TUNE_LAYERS layers are unfrozen,
    while the rest remain frozen.

    BatchNormalization layers remain frozen during fine-tuning
    for greater stability on the relatively small RFMiD dataset.
    """

    if pretrained_model_path is None:
        pretrained_model_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                "netrax_best.keras",
            )
        )

    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(
            f"Baseline model not found:\n"
            f"{pretrained_model_path}"
        )

    baseline = tf.keras.models.load_model(
        pretrained_model_path,
        compile=False,
    )

    # Locate the EfficientNet backbone.
    base_model = None

    for layer in baseline.layers:
        if layer.name == "efficientnetb0":
            base_model = layer
            break

    if base_model is None:
        raise ValueError(
            "Could not find the 'efficientnetb0' backbone "
            "inside the baseline model."
        )

    # Start with everything frozen.
    base_model.trainable = True

    total_layers = len(base_model.layers)

    fine_tune_start = max(
        0,
        total_layers - FINE_TUNE_LAYERS,
    )

    for index, layer in enumerate(base_model.layers):

        if index < fine_tune_start:
            layer.trainable = False

        else:
            # Keep BatchNorm frozen for stability.
            if isinstance(
                layer,
                layers.BatchNormalization,
            ):
                layer.trainable = False
            else:
                layer.trainable = True

    # Rebuild using the existing baseline weights and head.
    inputs = baseline.input
    outputs = baseline.output

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="NetraX_EfficientNetB0_FineTuned",
    )

    return model


def print_finetune_summary(model):

    trainable_params = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.trainable_weights
    )

    non_trainable_params = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.non_trainable_weights
    )

    print("=" * 70)
    print("NETRAX — FINE-TUNING MODEL")
    print("=" * 70)

    print("\nModel:", model.name)
    print("Output shape:", model.output_shape)

    print(
        "\nTrainable parameters:",
        trainable_params,
    )

    print(
        "Non-trainable parameters:",
        non_trainable_params,
    )

    print(
        "Total parameters:",
        trainable_params + non_trainable_params,
    )

    print(
        "\nFine-tuned EfficientNet layers:",
        FINE_TUNE_LAYERS,
    )

    print("\nTrainable EfficientNet layers:")

    for layer in model.layers:
        if layer.name == "efficientnetb0":

            trainable = [
                sublayer.name
                for sublayer in layer.layers
                if sublayer.trainable
            ]

            print(
                "Count:",
                len(trainable),
            )

            for name in trainable:
                print(" -", name)

            break


if __name__ == "__main__":

    model = build_finetune_model()

    print_finetune_summary(model)

    # Verify that the model can process one synthetic input.
    dummy_input = tf.zeros(
        (1, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=tf.float32,
    )

    output = model(
        dummy_input,
        training=False,
    )

    print("\nTest input shape:", dummy_input.shape)
    print("Test output shape:", output.shape)

    assert output.shape == (
        1,
        NUM_CLASSES,
    )

    print("\nFine-tuning model verification passed.")
