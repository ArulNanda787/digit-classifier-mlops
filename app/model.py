"""Load the trained Keras model and run predictions."""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "digit_cnn.keras"


def get_model_path() -> Path:
    """MODEL_PATH env var wins; otherwise use the Phase 1 output."""
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


class DigitPredictor:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_path(cls, path: Path) -> "DigitPredictor":
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run Phase 1: python -m src.train"
            )
        # Imported here so the API (and its tests) can start without
        # paying TensorFlow's import cost until a model is actually needed.
        from tensorflow import keras

        logger.info("Loading model from %s", path)
        return cls(keras.models.load_model(path))

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (10,), for a single image."""
        probs = self.model.predict(batch, verbose=0)
        return np.asarray(probs, dtype=np.float32)[0]
