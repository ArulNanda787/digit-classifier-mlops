import shutil
from pathlib import Path

import keras
import mlflow
import numpy as np
import tensorflow as tf
from keras import layers
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "mnist.npz"
MODEL_DIR = PROJECT_ROOT / "models"
CANDIDATE_DIR = MODEL_DIR / "candidates"
MODEL_PATH = MODEL_DIR / "digit_cnn.keras"  # the file the API loads

RANDOM_SEED = 42
EXPERIMENT_NAME = "MNIST-Digit-Classifier"


def build_cnn():
    """Good model: two conv blocks, learns shapes like strokes and curves."""
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_digit_cnn",
    )


def build_weak_mlp():
    """Deliberately bad model: no convolutions, one tiny hidden layer.

    Flattening throws away the 2-D layout of the image, and 16 neurons
    are too few to learn ten digit shapes well.
    """
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_weak_mlp",
    )


# Each entry becomes one MLflow run.
CONFIGS = [
    {
        "run_name": "cnn-baseline",
        "builder": build_cnn,
        "epochs": 5,
        "batch_size": 128,
        "learning_rate": 0.001,
        "train_samples": None,  # None = all 60,000
    },
    {
        "run_name": "weak-mlp",
        "builder": build_weak_mlp,
        "epochs": 1,
        "batch_size": 128,
        "learning_rate": 0.001,
        "train_samples": 5000,  # also starved of data
    },
]


def train_one(config, x_train, y_train, x_test, y_test):
    """Train one configuration inside its own MLflow run."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)  # same start for every run

    n = config["train_samples"] or len(x_train)
    x_tr, y_tr = x_train[:n], y_train[:n]

    with mlflow.start_run(run_name=config["run_name"]) as run:
        model = config["builder"]()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        mlflow.log_params(
            {
                "architecture": model.name,
                "trainable_params": model.count_params(),
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "learning_rate": config["learning_rate"],
                "training_samples": n,
                "test_samples": len(x_test),
                "random_seed": RANDOM_SEED,
                "optimizer": "Adam",
                "keras_version": keras.__version__,
                "tensorflow_version": tf.__version__,
            }
        )

        history = model.fit(
            x_tr,
            y_tr,
            validation_split=0.1,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                )
            ],
            verbose=1,
        )

        # Per-epoch curves, so MLflow can draw training charts.
        for epoch, (acc, val_acc) in enumerate(
            zip(history.history["accuracy"], history.history["val_accuracy"])
        ):
            mlflow.log_metric("train_accuracy", float(acc), step=epoch)
            mlflow.log_metric("val_accuracy", float(val_acc), step=epoch)

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        best_val_accuracy = float(max(history.history["val_accuracy"]))

        mlflow.log_metrics(
            {
                "best_validation_accuracy": best_val_accuracy,
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
            }
        )

        candidate_path = CANDIDATE_DIR / f"{config['run_name']}.keras"
        model.save(candidate_path)
        mlflow.log_artifact(str(candidate_path), artifact_path="model")

        print(
            f"[{config['run_name']}] val_acc={best_val_accuracy:.4f} "
            f"test_acc={test_accuracy:.4f}"
        )
        return {
            "run_id": run.info.run_id,
            "run_name": config["run_name"],
            "val_accuracy": best_val_accuracy,
            "test_accuracy": float(test_accuracy),
            "path": candidate_path,
        }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run: python -m src.data_preprocessing"
        )

    data = np.load(DATA_FILE)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = [
        train_one(cfg, x_train, y_train, x_test, y_test) for cfg in CONFIGS
    ]

    # Choose on validation accuracy, not test accuracy, so the test set
    # stays an unbiased final check.
    best = max(results, key=lambda r: r["val_accuracy"])
    shutil.copy(best["path"], MODEL_PATH)

    client = MlflowClient()
    for r in results:
        client.set_tag(
            r["run_id"],
            "selected_for_deployment",
            "true" if r is best else "false",
        )

    print("\nComparison")
    for r in results:
        mark = "  <- deployed" if r is best else ""
        print(
            f"  {r['run_name']:14s} val={r['val_accuracy']:.4f} "
            f"test={r['test_accuracy']:.4f}{mark}"
        )
    print(f"\nSaved best model to {MODEL_PATH}")
    print(f"Best run ID: {best['run_id']}")


if __name__ == "__main__":
    main()