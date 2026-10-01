from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "ml" / "scripts")
)

from v2_dataset import create_dataset, LABELS
from v2_loss import (
    masked_weighted_binary_crossentropy,
    calculate_masked_class_weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 384
NUM_CLASSES = 45
BATCH_SIZE = 16

EPOCHS = 20
LEARNING_RATE = 1e-5

# Fine-tune the upper portion of EfficientNetB0.
# This is deliberately conservative.
FINE_TUNE_LAST_N = 30

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL = (
    MODEL_DIR
    / "netrax_v21_best_auc.keras"
)

FINAL_MODEL = (
    MODEL_DIR
    / "netrax_v21_final.keras"
)

HISTORY_FILE = (
    MODEL_DIR
    / "netrax_v21_history.json"
)


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        name="image"
    )

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    )

    # --------------------------------------------------------
    # Start with everything frozen.
    # --------------------------------------------------------

    backbone.trainable = False

    # --------------------------------------------------------
    # Fine-tune only the final N layers.
    # BatchNorm remains frozen because retinal datasets
    # can be sensitive to small-batch BN statistics.
    # --------------------------------------------------------

    if FINE_TUNE_LAST_N > 0:

        for layer in backbone.layers[
            -FINE_TUNE_LAST_N:
        ]:

            if isinstance(
                layer,
                layers.BatchNormalization
            ):

                layer.trainable = False

            else:

                layer.trainable = True

    x = backbone(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(
        x
    )

    x = layers.BatchNormalization()(
        x
    )

    x = layers.Dropout(
        0.35
    )(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.25
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="sigmoid",
        name="predictions"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="NetraX_EfficientNetB0_V2_1"
    )

    return model


# ============================================================
# PREDICT DATASET
# ============================================================

def predict_dataset(
    model,
    dataset
):

    all_predictions = []
    all_targets = []
    all_masks = []

    for images, labels in dataset:

        predictions = model(
            images,
            training=False
        )

        all_predictions.append(
            predictions.numpy()
        )

        all_targets.append(
            labels["targets"].numpy()
        )

        all_masks.append(
            labels["mask"].numpy()
        )

    return (
        np.concatenate(
            all_predictions,
            axis=0
        ),
        np.concatenate(
            all_targets,
            axis=0
        ),
        np.concatenate(
            all_masks,
            axis=0
        ),
    )


# ============================================================
# MACRO ROC-AUC
# ============================================================

def masked_auc(
    predictions,
    targets,
    masks
):

    aucs = []

    for i in range(
        NUM_CLASSES
    ):

        known = (
            masks[:, i]
            > 0.5
        )

        y_true = targets[
            known,
            i
        ]

        y_pred = predictions[
            known,
            i
        ]

        if len(y_true) == 0:

            continue

        if len(
            np.unique(y_true)
        ) < 2:

            continue

        metric = tf.keras.metrics.AUC(
            curve="ROC"
        )

        metric.update_state(
            y_true,
            y_pred
        )

        aucs.append(
            float(
                metric.result().numpy()
            )
        )

    if not aucs:

        return float("nan")

    return float(
        np.mean(aucs)
    )


# ============================================================
# CUSTOM MODEL WITH MASKED LOSS
# ============================================================

class MaskedLossModel(
    tf.keras.Model
):

    def __init__(
        self,
        base_model,
        class_weights
    ):

        super().__init__()

        self.base_model = (
            base_model
        )

        self.class_weights = (
            tf.constant(
                class_weights,
                dtype=tf.float32
            )
        )

        self.loss_tracker = (
            tf.keras.metrics.Mean(
                name="loss"
            )
        )

    @property
    def metrics(self):

        return [
            self.loss_tracker
        ]

    def train_step(
        self,
        data
    ):

        images, labels = data

        targets = labels[
            "targets"
        ]

        mask = labels[
            "mask"
        ]

        with tf.GradientTape() as tape:

            predictions = (
                self.base_model(
                    images,
                    training=True
                )
            )

            loss = (
                masked_weighted_binary_crossentropy(
                    targets,
                    predictions,
                    mask,
                    self.class_weights
                )
            )

        gradients = tape.gradient(
            loss,
            self.base_model.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.base_model.trainable_variables
            )
        )

        self.loss_tracker.update_state(
            loss
        )

        return {
            "loss":
                self.loss_tracker.result()
        }

    def test_step(
        self,
        data
    ):

        images, labels = data

        targets = labels[
            "targets"
        ]

        mask = labels[
            "mask"
        ]

        predictions = (
            self.base_model(
                images,
                training=False
            )
        )

        loss = (
            masked_weighted_binary_crossentropy(
                targets,
                predictions,
                mask,
                self.class_weights
            )
        )

        self.loss_tracker.update_state(
            loss
        )

        return {
            "loss":
                self.loss_tracker.result()
        }


# ============================================================
# RFMiD VALIDATION CALLBACK
# ============================================================

class RFMiDValidationCallback(
    tf.keras.callbacks.Callback
):

    def __init__(
        self,
        val_dataset
    ):

        super().__init__()

        self.val_dataset = (
            val_dataset
        )

        self.val_auc_history = []
        self.val_loss_history = []

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        logs = logs or {}

        predictions, targets, masks = (
            predict_dataset(
                self.model.base_model,
                self.val_dataset
            )
        )

        loss = (
            masked_weighted_binary_crossentropy(
                tf.convert_to_tensor(
                    targets
                ),
                tf.convert_to_tensor(
                    predictions
                ),
                tf.convert_to_tensor(
                    masks
                ),
                self.model.class_weights
            )
        )

        auc = masked_auc(
            predictions,
            targets,
            masks
        )

        loss_value = float(
            loss.numpy()
        )

        self.val_loss_history.append(
            loss_value
        )

        self.val_auc_history.append(
            auc
        )

        logs["val_loss"] = (
            loss_value
        )

        logs["val_auc"] = (
            auc
        )

        print()
        print(
            "RFMiD validation - "
            f"loss: {loss_value:.5f} - "
            f"macro_auc: {auc:.5f}"
        )


# ============================================================
# BEST AUC MODEL SAVER
# ============================================================

class BestAUCModelSaver(
    tf.keras.callbacks.Callback
):

    def __init__(
        self,
        filepath
    ):

        super().__init__()

        self.filepath = (
            filepath
        )

        self.best = -np.inf

        self.best_epoch = None

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        logs = logs or {}

        value = logs.get(
            "val_auc"
        )

        if value is None:

            return

        if not np.isfinite(value):

            return

        if value > self.best:

            self.best = float(
                value
            )

            self.best_epoch = (
                epoch + 1
            )

            self.model.base_model.save(
                self.filepath
            )

            print()
            print(
                "Saved best V2.1 model: "
                f"epoch={self.best_epoch} "
                f"val_auc={self.best:.5f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "NETRAX V2.1 TRAINING"
    )
    print("=" * 70)

    print()
    print(
        "Strategy:"
    )

    print(
        "- Combined RFMiD + ODIR training"
    )

    print(
        "- RFMiD validation only"
    )

    print(
        "- EfficientNetB0 ImageNet initialization"
    )

    print(
        f"- Fine-tune last {FINE_TUNE_LAST_N} layers"
    )

    print(
        "- BatchNorm layers frozen"
    )

    print(
        "- Masked weighted BCE"
    )

    print(
        "- Best checkpoint selected by RFMiD validation macro ROC-AUC"
    )

    print()

    # ========================================================
    # DATASETS
    # ========================================================

    print(
        "Loading datasets..."
    )

    train_dataset, train_df = (
        create_dataset(
            split="train",
            source=None,
            batch_size=BATCH_SIZE,
            training=True,
            shuffle=True
        )
    )

    val_dataset, val_df = (
        create_dataset(
            split="validation",
            source="RFMiD",
            batch_size=BATCH_SIZE,
            training=False,
            shuffle=False
        )
    )

    print()
    print(
        "Training records:",
        len(train_df)
    )

    print(
        "RFMiD validation:",
        len(val_df)
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    print()
    print(
        "Calculating class weights..."
    )

    targets = (
        train_df[
            LABELS
        ]
        .values
        .astype(
            np.float32
        )
    )

    masks = (
        train_df[
            [
                f"{label}_known"
                for label in LABELS
            ]
        ]
        .values
        .astype(
            np.float32
        )
    )

    class_weights = (
        calculate_masked_class_weights(
            targets,
            masks,
            max_weight=10.0
        )
    )

    print()
    print(
        "Class weights:"
    )

    for label, weight in zip(
        LABELS,
        class_weights
    ):

        print(
            "{:<6} {:.3f}".format(
                label,
                weight
            )
        )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Building V2.1 model..."
    )

    base_model = build_model()

    print()
    print(
        "Trainable layers:"
    )

    trainable_count = 0

    for layer in (
        base_model.layers
    ):

        if layer.trainable:

            trainable_count += 1

            print(
                f"  {layer.name}"
            )

    print()
    print(
        "Trainable layer count:",
        trainable_count
    )

    print()
    print(
        "Model summary:"
    )

    base_model.summary()

    # ========================================================
    # CUSTOM TRAINING MODEL
    # ========================================================

    model = MaskedLossModel(
        base_model,
        class_weights
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )
    )

    # ========================================================
    # CALLBACKS
    # ========================================================

    validation_callback = (
        RFMiDValidationCallback(
            val_dataset
        )
    )

    best_callback = (
        BestAUCModelSaver(
            BEST_MODEL
        )
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            mode="max",
            restore_best_weights=False,
            verbose=1
        )
    )

    reduce_lr = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            mode="max",
            verbose=1
        )
    )

    callbacks = [
        validation_callback,
        best_callback,
        early_stopping,
        reduce_lr,
    ]

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STARTING V2.1 TRAINING"
    )
    print("=" * 70)

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    print()
    print(
        "Saving final model..."
    )

    model.base_model.save(
        FINAL_MODEL
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_serializable = {}

    for key, values in (
        history.history.items()
    ):

        history_serializable[
            key
        ] = [
            (
                float(v)
                if np.isfinite(v)
                else None
            )
            for v in values
        ]

    history_serializable[
        "rfmid_val_auc"
    ] = (
        validation_callback
        .val_auc_history
    )

    history_serializable[
        "rfmid_val_loss"
    ] = (
        validation_callback
        .val_loss_history
    )

    history_serializable[
        "best_auc"
    ] = (
        float(
            best_callback.best
        )
        if np.isfinite(
            best_callback.best
        )
        else None
    )

    history_serializable[
        "best_auc_epoch"
    ] = (
        best_callback.best_epoch
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history_serializable,
            f,
            indent=2
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "V2.1 TRAINING COMPLETE"
    )
    print("=" * 70)

    print()

    print(
        "Best AUC:"
    )

    print(
        best_callback.best
    )

    print()

    print(
        "Best epoch:"
    )

    print(
        best_callback.best_epoch
    )

    print()

    print(
        "Best model:"
    )

    print(
        BEST_MODEL
    )

    print()

    print(
        "Final model:"
    )

    print(
        FINAL_MODEL
    )

    print()

    print(
        "History:"
    )

    print(
        HISTORY_FILE
    )


if __name__ == "__main__":

    main()