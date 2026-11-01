"""FastAPI app: prediction API, drawing-canvas frontend and Prometheus metrics."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from starlette.routing import Match

from app.model import DigitPredictor, get_model_path
from app.preprocessing import ImagePreprocessingError, preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digit-api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# ---------------------------------------------------------------------------
# Prometheus metrics (scraped from GET /metrics)
# ---------------------------------------------------------------------------
REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests, by method, endpoint and status code",
    ["method", "endpoint", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "predictions_total", "Successful predictions, by predicted digit", ["digit"]
)
CONFIDENCE = Histogram(
    "prediction_confidence",
    "Model confidence for the predicted digit",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total", "Rejected prediction requests, by reason", ["reason"]
)
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded, otherwise 0")

# Create every label up front so dashboards show 0 instead of "no data".
for _digit in range(10):
    PREDICTIONS.labels(digit=str(_digit))
for _reason in ("invalid_image", "unsupported_type", "too_large", "model_unavailable"):
    PREDICTION_ERRORS.labels(reason=_reason)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = DigitPredictor.from_path(get_model_path())
        MODEL_LOADED.set(1)
        logger.info("Model loaded.")
    except Exception:
        logger.exception("Model failed to load; /predict will return 503.")
        app.state.predictor = None
        MODEL_LOADED.set(0)
    yield


app = FastAPI(
    title="MNIST Digit Classifier",
    version="1.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _endpoint_label(request: Request) -> str:
    """Route template (e.g. "/predict"), so random URLs don't create new series."""
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", "other")
    return "unmatched"


@app.middleware("http")
async def track_requests(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    endpoint = _endpoint_label(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(
            method=request.method, endpoint=endpoint, status=str(status)
        ).inc()


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


def get_predictor(request: Request) -> DigitPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        PREDICTION_ERRORS.labels(reason="model_unavailable").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return predictor


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    loaded = getattr(request.app.state, "predictor", None) is not None
    return {"status": "ok", "model_loaded": loaded}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    predictor: DigitPredictor = Depends(get_predictor),
):
    if file.content_type and not file.content_type.startswith("image/"):
        PREDICTION_ERRORS.labels(reason="unsupported_type").inc()
        raise HTTPException(
            status_code=415, detail="Upload an image file (PNG or JPEG)."
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        PREDICTION_ERRORS.labels(reason="too_large").inc()
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB.")

    try:
        batch = preprocess_image(data)
    except ImagePreprocessingError as exc:
        PREDICTION_ERRORS.labels(reason="invalid_image").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probs = predictor.predict(batch)
    digit = int(probs.argmax())
    confidence = float(probs[digit])

    PREDICTIONS.labels(digit=str(digit)).inc()
    CONFIDENCE.observe(confidence)

    return PredictionResponse(
        digit=digit,
        confidence=round(confidence, 4),
        probabilities=[round(float(p), 4) for p in probs],
    )
