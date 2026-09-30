from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from app.services.disease_metadata import DISEASE_METADATA


# ============================================================
# NetraX Inference Service
# ============================================================

IMAGE_SIZE = 384
NUM_CLASSES = 45

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
ML_ROOT = PROJECT_ROOT / "ml"

MODEL_PATH = ML_ROOT / "models" / "netrax_finetuned_best.keras"
THRESHOLD_PATH = (
    ML_ROOT
    / "evaluation"
    / "finetuned_per_label_thresholds.json"
)


LABELS = [
    "DR",
    "ARMD",
    "MH",
    "DN",
    "MYA",
    "BRVO",
    "TSLN",
    "ERM",
    "LS",
    "MS",
    "CSR",
    "ODC",
    "CRVO",
    "TV",
    "AH",
    "ODP",
    "ODE",
    "ST",
    "AION",
    "PT",
    "RT",
    "RS",
    "CRS",
    "EDN",
    "RPEC",
    "MHL",
    "RP",
    "CWS",
    "CB",
    "ODPM",
    "PRH",
    "MNF",
    "HR",
    "CRAO",
    "TD",
    "CME",
    "PTCR",
    "CF",
    "VH",
    "MCA",
    "VS",
    "BRAO",
    "PLQ",
    "HPED",
    "CL",
]


class NetraXInference:
    """
    Loads the NetraX model and validation-derived thresholds
    once and exposes image prediction functionality.
    """

    def __init__(self):
        self.model = None
        self.thresholds = None

    def load(self):
        """Load model and threshold configuration."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"NetraX model not found: {MODEL_PATH}"
            )

        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError(
                f"NetraX threshold file not found: "
                f"{THRESHOLD_PATH}"
            )

        print(f"Loading NetraX model: {MODEL_PATH}")

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        if self.model.output_shape[-1] != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} model outputs, "
                f"got {self.model.output_shape[-1]}"
            )

        with open(
            THRESHOLD_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            threshold_data = json.load(file)

        self.thresholds = np.asarray(
            [
                float(threshold_data[label]["threshold"])
                for label in LABELS
            ],
            dtype=np.float32,
        )

        if len(self.thresholds) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} thresholds, "
                f"got {len(self.thresholds)}"
            )

        print("NetraX model loaded successfully.")
        print(
            f"Model outputs: {NUM_CLASSES} disease probabilities"
        )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Convert an uploaded PIL image into the exact input
        representation expected by the trained model.
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = tf.image.resize_with_pad(
            tf.convert_to_tensor(
                np.asarray(image),
                dtype=tf.float32,
            ),
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        image = tf.clip_by_value(
            image,
            0.0,
            255.0,
        )

        image = tf.expand_dims(
            image,
            axis=0,
        )

        return image.numpy()

    def predict(self, image: Image.Image) -> dict:
        """Run NetraX inference on a retinal image."""

        if self.model is None or self.thresholds is None:
            raise RuntimeError(
                "NetraX inference service has not been loaded."
            )

        input_tensor = self.preprocess(image)

        probabilities = self.model.predict(
            input_tensor,
            verbose=0,
        )[0]

        predictions = probabilities >= self.thresholds

        detected_conditions = []

        for index, label in enumerate(LABELS):
            probability = float(probabilities[index])
            threshold = float(self.thresholds[index])

            detected_conditions.append(
                {
                    "label": label,
                    "name": DISEASE_METADATA[label]["name"],
                    "description": DISEASE_METADATA[label]["description"],
                    "probability": probability,
                    "confidence_percent": round(
                        probability * 100,
                        2,
                    ),
                    "threshold": threshold,
                    "detected": bool(predictions[index]),
                }
            )

        detected_conditions.sort(
            key=lambda item: item["probability"],
            reverse=True,
        )

        detected = [
            item
            for item in detected_conditions
            if item["detected"]
        ]

        return {
            "detected_conditions": detected,
            "all_predictions": detected_conditions,
        }


netrax_inference = NetraXInference()