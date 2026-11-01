"""Turn an uploaded or drawn image into the input format the CNN was trained on.

MNIST digits are white strokes on a black background, scaled into a 20x20 box,
centered by center of mass inside a 28x28 frame, with pixels in [0, 1].
This module reproduces those steps so canvas drawings and photos match the
training data as closely as possible.
"""

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = 28
DIGIT_BOX = 20
INK_THRESHOLD = 30  # pixel values (0-255) below this count as background


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot be turned into a model input."""


def _load_grayscale(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImagePreprocessingError("File is not a readable image.") from exc

    # Put transparent images on a white background before converting.
    if img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.alpha_composite(img)
        img = background

    return np.asarray(img.convert("L"), dtype=np.float32)


def _shift_with_zeros(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.zeros_like(arr)
    h, w = arr.shape
    src_y = slice(max(0, -dy), min(h, h - dy))
    dst_y = slice(max(0, dy), min(h, h + dy))
    src_x = slice(max(0, -dx), min(w, w - dx))
    dst_x = slice(max(0, dx), min(w, w + dx))
    shifted[dst_y, dst_x] = arr[src_y, src_x]
    return shifted


def preprocess_array(gray: np.ndarray) -> np.ndarray:
    """Convert a 2-D grayscale array (0-255) to a (1, 28, 28, 1) float array."""
    if gray.ndim != 2:
        raise ImagePreprocessingError("Expected a 2-D grayscale image.")

    gray = gray.astype(np.float32)

    # Invert dark-on-light images (paper, white canvas) to MNIST's
    # light-on-dark style, judging by the average border brightness.
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if border.mean() > 127:
        gray = 255.0 - gray

    gray[gray < INK_THRESHOLD] = 0.0
    ink = np.argwhere(gray > 0)
    if ink.size == 0:
        raise ImagePreprocessingError("No digit found in the image.")

    # Crop to the digit's bounding box.
    (y0, x0), (y1, x1) = ink.min(axis=0), ink.max(axis=0) + 1
    digit = gray[y0:y1, x0:x1]

    # Scale the longest side to 20 px, keeping the aspect ratio.
    h, w = digit.shape
    scale = DIGIT_BOX / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = Image.fromarray(digit.astype(np.uint8)).resize(
        (new_w, new_h), Image.Resampling.LANCZOS
    )

    # Paste into the middle of a 28x28 black frame.
    frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    top, left = (IMAGE_SIZE - new_h) // 2, (IMAGE_SIZE - new_w) // 2
    frame[top : top + new_h, left : left + new_w] = np.asarray(
        resized, dtype=np.float32
    )

    # Shift so the center of mass sits at the frame's center, as in MNIST.
    total = frame.sum()
    if total > 0:
        ys, xs = np.indices(frame.shape)
        cy, cx = (ys * frame).sum() / total, (xs * frame).sum() / total
        frame = _shift_with_zeros(
            frame,
            int(round(IMAGE_SIZE / 2 - cy)),
            int(round(IMAGE_SIZE / 2 - cx)),
        )

    frame = np.clip(frame / 255.0, 0.0, 1.0)
    return frame.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes (PNG, JPEG, ...) to a model-ready array."""
    if not image_bytes:
        raise ImagePreprocessingError("Empty file.")
    return preprocess_array(_load_grayscale(image_bytes))
