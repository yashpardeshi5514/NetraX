import os
import json
import tensorflow as tf

from dataset import (
    BATCH_SIZE,
    NUM_CLASSES,
    create_dataset,
    calculate_class_weights,
)
from losses import weighted_binary_crossentropy
from model import build_model


# ============================================================
# NetraX — Actual Training Pipeline
# ============================================================

SEED = 42

IMAGE_SIZE = 384
LEARNING_RATE = 1e-3

EPOCHS = 15

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation",
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "netrax_best.keras",
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "netrax_final.keras",
)

HISTORY_PATH = os.path.join(
    EVALUATION_DIR,
    "training_history.json",
)

tf.random.set_seed(SEED)


def build_training_components():

    print("Loading datasets...")

    train_dataset = create_dataset(
        "train",
        batch_size=BATCH_SIZE,
        training=True,
    )

    validation_dataset = create_dataset(
        "validation",
        batch_size=BATCH_SIZE,
        training=False,
    )

    print("Calculating class weights...")

    class_weights_series = calculate_class_weights(
        "train",
        max_weight=10.0,
    )

    class_weights = tf.constant(
        class_weights_series.values,
        dtype=tf.float32,
    )

    if class_weights.shape[0] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} class weights, "
            f"got {class_weights.shape[0]}"
        )

    print("Building model...")

    model = build_model(
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
    )

    loss_fn = weighted_binary_crossentropy(
        class_weights
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy"
            ),
            tf.keras.metrics.AUC(
                name="macro_auc",
                multi_label=True,
                num_labels=NUM_CLASSES,
            ),
        ],
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            os.path.join(
                EVALUATION_DIR,
                "training_log.csv",
            )
        ),
    ]

    return (
        train_dataset,
        validation_dataset,
        model,
        callbacks,
        class_weights,
    )


def train():

    print("=" * 70)
    print("NETRAX — ACTUAL MODEL TRAINING")
    print("=" * 70)

    print("\nConfiguration")
    print("-" * 70)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Classes:", NUM_CLASSES)
    print("Maximum epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Best model:", BEST_MODEL_PATH)

    (
        train_dataset,
        validation_dataset,
        model,
        callbacks,
        class_weights,
    ) = build_training_components()

    print("\nModel summary")
    print("-" * 70)

    model.summary()

    print("\nClass-weight range:")
    print(
        f"{float(tf.reduce_min(class_weights)):.2f}"
        f" - "
        f"{float(tf.reduce_max(class_weights)):.2f}"
    )

    print("\nStarting training...")
    print("=" * 70)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nTraining finished.")

    print("\nSaving final model...")

    model.save(
        FINAL_MODEL_PATH
    )

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history_data,
            file,
            indent=2,
        )

    print("\nSaved:")
    print(
        "Best model:",
        BEST_MODEL_PATH,
    )
    print(
        "Final model:",
        FINAL_MODEL_PATH,
    )
    print(
        "Training history:",
        HISTORY_PATH,
    )

    print("\nBest validation loss:")

    best_epoch = min(
        range(len(history.history["val_loss"])),
        key=lambda index:
        history.history["val_loss"][index],
    )

    print(
        f"Epoch {best_epoch + 1}: "
        f"{history.history['val_loss'][best_epoch]:.6f}"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    train()