# MNIST Digit Classifier — MLOps project

## Requirements
- Python 3.11
- pip
- Internet connection for the initial MNIST download

## Setup (macOS / Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Phase 1 — Run the data-preparation step

```bash
python -m src.data_preprocessing
```

This downloads MNIST through Keras, normalizes image pixels to [0, 1], adds the CNN channel dimension, and writes `data/processed/mnist.npz`.

## Phase 1 — Train and evaluate

```bash
python -m src.train
```

The trained model is saved as `models/digit_cnn.keras`. Training parameters, validation metrics, test metrics, and the model artifact are logged to the local MLflow tracking store in `mlruns/`.

## Phase 1 — View MLflow

In another terminal, from the project root:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000.

## Phase 1 — Expected outputs

- `data/processed/mnist.npz`
- `models/digit_cnn.keras`
- `mlruns/`

Accuracy is a measured result, not a guaranteed value. Record the actual test accuracy from your run.

## Phase 2 — Prediction application

### Install the new dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the tests

```bash
python -m pytest -v
```

The API tests use a fake model, so they run without TensorFlow loading. `test_real_model_on_mnist_samples` uses `models/digit_cnn.keras` and 50 MNIST test images, and is skipped when either file is missing.

### Start the app

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Drawing canvas: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Set `MODEL_PATH` to load a different model file, for example `MODEL_PATH=/path/to/model.keras uvicorn app.main:app`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Drawing canvas frontend |
| GET | `/health` | `{"status": "ok", "model_loaded": true}` |
| POST | `/predict` | Multipart image upload (`file` field) → `{"digit", "confidence", "probabilities"}` |

Errors: 400 for an unreadable or blank image, 413 if larger than 5 MB, 415 for non-image uploads, 422 if no file is sent, 503 if the model isn't loaded.

### Test with sample images

```bash
python scripts/export_samples.py --count 20      # writes samples/*.png
curl -F "file=@samples/000_label7.png" http://127.0.0.1:8000/predict
python scripts/predict_samples.py                 # predicts every sample, prints accuracy
```

Each sample is exported twice: MNIST-style (white on black, 28×28) and `_paper` (black on white, 360×360) to check the inversion and resizing steps.

### Preprocessing

`app/preprocessing.py` converts any input into the format the model was trained on: grayscale, inverted if the background is light, cropped to the digit, scaled into a 20×20 box, centered by center of mass in a 28×28 frame, and scaled to [0, 1].
