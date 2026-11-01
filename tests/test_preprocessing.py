import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.preprocessing import ImagePreprocessingError, preprocess_image


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _drawn_one(bg: int, fg: int, size: int = 280) -> bytes:
    """A thick vertical stroke, roughly like a hand-drawn '1'."""
    img = Image.new("L", (size, size), color=bg)
    ImageDraw.Draw(img).line(
        [(size * 0.6, size * 0.15), (size * 0.6, size * 0.85)],
        fill=fg,
        width=size // 14,
    )
    return _png(img)


def test_output_shape_and_range():
    arr = preprocess_image(_drawn_one(bg=0, fg=255))
    assert arr.shape == (1, 28, 28, 1)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0
    assert arr.max() > 0.5


def test_digit_is_centered():
    arr = preprocess_image(_drawn_one(bg=0, fg=255))[0, :, :, 0]
    ys, xs = np.indices(arr.shape)
    cy = (ys * arr).sum() / arr.sum()
    cx = (xs * arr).sum() / arr.sum()
    assert abs(cy - 14) <= 1.5 and abs(cx - 14) <= 1.5
    # Background corners stay black.
    assert arr[0, 0] == arr[0, -1] == arr[-1, 0] == arr[-1, -1] == 0


def test_dark_on_light_is_inverted():
    white_bg = preprocess_image(_drawn_one(bg=255, fg=0))
    black_bg = preprocess_image(_drawn_one(bg=0, fg=255))
    assert np.allclose(white_bg, black_bg, atol=0.05)


def test_transparent_png():
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(img).line([(100, 30), (100, 170)], fill=(0, 0, 0, 255), width=16)
    arr = preprocess_image(_png(img))
    assert arr.max() > 0.5


def test_blank_image_rejected():
    with pytest.raises(ImagePreprocessingError):
        preprocess_image(_png(Image.new("L", (280, 280), color=0)))


def test_non_image_rejected():
    with pytest.raises(ImagePreprocessingError):
        preprocess_image(b"definitely not an image")


def test_empty_bytes_rejected():
    with pytest.raises(ImagePreprocessingError):
        preprocess_image(b"")
