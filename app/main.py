"""FastAPI app: prediction API plus the drawing-canvas frontend."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.model import DigitPredictor, get_model_path
from app.preprocessing import ImagePreprocessingError, preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digit-api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = DigitPredictor.from_path(get_model_path())
        logger.info("Model loaded.")
    except Exception:
        logger.exception("Model failed to load; /predict will return 503.")
        app.state.predictor = None
    yield


app = FastAPI(
    title="MNIST Digit Classifier",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


def get_predictor(request: Request) -> DigitPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return predictor


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    loaded = getattr(request.app.state, "predictor", None) is not None
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    predictor: DigitPredictor = Depends(get_predictor),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415, detail="Upload an image file (PNG or JPEG)."
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB.")

    try:
        batch = preprocess_image(data)
    except ImagePreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probs = predictor.predict(batch)
    digit = int(probs.argmax())
    return PredictionResponse(
        digit=digit,
        confidence=round(float(probs[digit]), 4),
        probabilities=[round(float(p), 4) for p in probs],
    )
