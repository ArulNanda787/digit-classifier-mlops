"""API tests.

Unit tests swap in a fake model so they run fast and don't need TensorFlow
or the trained model file. The integration test at the bottom uses the real
model and is skipped if models/digit_cnn.keras is missing.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app, get_predictor
from app.model import get_model_path


class FakePredictor:
    def predict(self, batch):
        assert batch.shape == (1, 28, 28, 1)
        probs = np.full(10, 0.01, dtype=np.float32)
        probs[7] = 0.91
        return probs


def _digit_png() -> bytes:
    img = Image.new("L", (280, 280), color=0)
    ImageDraw.Draw(img).line([(140, 40), (140, 240)], fill=255, width=20)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    # No "with" block: the startup hook (which loads the real model) is skipped.
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_index_serves_canvas(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "<canvas" in res.text


def test_predict_returns_digit(client):
    res = client.post(
        "/predict", files={"file": ("digit.png", _digit_png(), "image/png")}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["digit"] == 7
    assert body["confidence"] == pytest.approx(0.91, abs=1e-3)
    assert len(body["probabilities"]) == 10


def test_predict_rejects_non_image(client):
    res = client.post(
        "/predict", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert res.status_code == 415


def test_predict_rejects_corrupt_image(client):
    res = client.post(
        "/predict", files={"file": ("bad.png", b"not really a png", "image/png")}
    )
    assert res.status_code == 400


def test_predict_rejects_blank_canvas(client):
    buf = io.BytesIO()
    Image.new("L", (280, 280), color=0).save(buf, format="PNG")
    res = client.post(
        "/predict", files={"file": ("blank.png", buf.getvalue(), "image/png")}
    )
    assert res.status_code == 400
    assert "No digit" in res.json()["detail"]


def test_predict_requires_file(client):
    assert client.post("/predict").status_code == 422


def test_predict_503_when_model_missing():
    app.dependency_overrides.clear()
    app.state.predictor = None
    res = TestClient(app).post(
        "/predict", files={"file": ("digit.png", _digit_png(), "image/png")}
    )
    assert res.status_code == 503


@pytest.mark.skipif(not get_model_path().exists(), reason="trained model not found")
def test_real_model_on_mnist_samples():
    """Real model + real MNIST test images exported to PNG."""
    data_file = get_model_path().parents[1] / "data" / "processed" / "mnist.npz"
    if not data_file.exists():
        pytest.skip("processed MNIST data not found")

    data = np.load(data_file)
    images, labels = data["x_test"][:50], data["y_test"][:50]

    app.dependency_overrides.clear()
    with TestClient(app) as real_client:  # runs startup, loads the model
        correct = 0
        for img, label in zip(images, labels):
            buf = io.BytesIO()
            Image.fromarray((img[:, :, 0] * 255).astype(np.uint8)).save(buf, "PNG")
            res = real_client.post(
                "/predict", files={"file": ("d.png", buf.getvalue(), "image/png")}
            )
            assert res.status_code == 200
            correct += int(res.json()["digit"] == int(label))

    assert correct / len(labels) >= 0.9
