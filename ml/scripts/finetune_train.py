from pathlib import Path
import json

import tensorflow as tf

from dataset import (
    create_dataset,
    calculate_class_weights,
)
from losses import weighted_binary_crossentropy
from finetune_model import build_finetune_model


# ============================================================
# NetraX — EfficientNet Fine-Tuning Training
# ============================================================

IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_CLASSES = 45
MAX_EPOCHS = 10
LEARNING_RATE = 1e-5

# Resolve paths from this script's location.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "evaluation"

BEST_MODEL_PATH = MODEL_DIR / "netrax_finetuned_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "netrax_finetuned_final.keras"
HISTORY_PATH = EVAL_DIR / "finetuning_history.json"
LOG_PATH = EVAL_DIR / "finetuning_log.csv"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NetraX — EfficientNet Fine-Tuning")
    print("=" * 70)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Best model:   {BEST_MODEL_PATH}")
    print(f"Final model:  {FINAL_MODEL_PATH}")

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------
    print("\nLoading training dataset...")

    train_ds = create_dataset(
        "train",
        batch_size=BATCH_SIZE,
        training=True,
    )

    print("Loading validation dataset...")

    val_ds = create_dataset(
        "validation",
        batch_size=BATCH_SIZE,
        training=False,
    )

    # ---------------------------------------------------------
    # Calculate class weights from TRAINING SET ONLY
    # ---------------------------------------------------------
    class_weights = calculate_class_weights(
        "train",
        max_weight=10.0,
    )

    print("\nClass-weight range:")
    print(
        f"min={float(class_weights.min()):.3f}, "
        f"max={float(class_weights.max()):.3f}"
    )

    # ---------------------------------------------------------
    # Build fine-tuning model
    # ---------------------------------------------------------
    print("\nBuilding fine-tuning model...")

    model = build_finetune_model()

    print("\nModel:", model.name)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} outputs, "
            f"but model has {model.output_shape[-1]}."
        )

    trainable_params = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.trainable_weights
    )

    non_trainable_params = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.non_trainable_weights
    )

    print(f"Trainable parameters:     {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(
        f"Total parameters:         "
        f"{trainable_params + non_trainable_params:,}"
    )

    # ---------------------------------------------------------
    # Compile
    #
    # TensorFlow's AUC metric does not accept an "average"
    # argument in this installed version. With multi_label=True,
    # AUC maintains one score per label and aggregates them
    # according to TensorFlow's multi-label implementation.
    # ---------------------------------------------------------
    loss_fn = weighted_binary_crossentropy(class_weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
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

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(LOG_PATH),
            append=False,
        ),
    ]

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("Starting fine-tuning")
    print("=" * 70)
    print(f"Image size:       {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Maximum epochs:   {MAX_EPOCHS}")
    print(f"Learning rate:    {LEARNING_RATE}")
    print("Training augmentation: enabled")
    print("Validation augmentation: disabled")
    print("Test set: NOT used during training")
    print("=" * 70)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ---------------------------------------------------------
    # Save final model
    # ---------------------------------------------------------
    model.save(FINAL_MODEL_PATH)

    # ---------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=2)

    # ---------------------------------------------------------
    # Determine best epoch
    # ---------------------------------------------------------
    val_losses = history.history["val_loss"]

    best_epoch_index = min(
        range(len(val_losses)),
        key=lambda index: val_losses[index],
    )

    best_epoch = best_epoch_index + 1
    best_val_loss = val_losses[best_epoch_index]

    print("\n" + "=" * 70)
    print("Fine-tuning complete")
    print("=" * 70)
    print(f"Best epoch:      {best_epoch}")
    print(f"Best val loss:   {best_val_loss:.6f}")
    print(f"Best model:      {BEST_MODEL_PATH}")
    print(f"Final model:     {FINAL_MODEL_PATH}")
    print(f"History:         {HISTORY_PATH}")
    print(f"Training log:    {LOG_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
