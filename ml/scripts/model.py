import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0


IMAGE_SIZE = 384
NUM_CLASSES = 45


def build_model(
    image_size=IMAGE_SIZE,
    num_classes=NUM_CLASSES
):

    inputs = layers.Input(
        shape=(image_size, image_size, 3),
        name="retinal_image"
    )

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(
            image_size,
            image_size,
            3
        )
    )

    # Stage 1:
    # Freeze pretrained feature extractor.
    base_model.trainable = False

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.35)(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(
        num_classes,
        activation="sigmoid",
        name="disease_probabilities"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="NetraX_EfficientNetB0"
    )

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()

    print("\nOutput shape:")
    print(model.output_shape)