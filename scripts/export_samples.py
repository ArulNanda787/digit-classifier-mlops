"""Export MNIST test images as PNG files for manual and scripted testing.

Writes samples/<index>_label<digit>.png (white on black, like MNIST) and a
black-on-white, upscaled copy samples/<index>_label<digit>_paper.png that
mimics a photo of a digit on paper.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "mnist.npz"
OUT_DIR = PROJECT_ROOT / "samples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit(f"{DATA_FILE} not found. Run: python -m src.data_preprocessing")

    data = np.load(DATA_FILE)
    images, labels = data["x_test"], data["y_test"]
    OUT_DIR.mkdir(exist_ok=True)

    for i in range(args.count):
        label = int(labels[i])
        img = Image.fromarray((images[i, :, :, 0] * 255).astype(np.uint8))
        img.save(OUT_DIR / f"{i:03d}_label{label}.png")

        paper = ImageOps.invert(img).resize((280, 280), Image.Resampling.BICUBIC)
        paper = ImageOps.expand(paper, border=40, fill=255)
        paper.save(OUT_DIR / f"{i:03d}_label{label}_paper.png")

    print(f"Wrote {args.count * 2} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
