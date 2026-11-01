"""Send every PNG in samples/ to the running API and report accuracy.

The true label is read from the file name (..._label<digit>...).
Usage: python scripts/predict_samples.py [--url http://127.0.0.1:8000]
"""

import argparse
import re
from pathlib import Path

import httpx

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    files = sorted(SAMPLES_DIR.glob("*.png"))
    if not files:
        raise SystemExit("No samples found. Run: python scripts/export_samples.py")

    correct = 0
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for path in files:
            match = re.search(r"label(\d)", path.name)
            expected = int(match.group(1)) if match else None
            res = client.post(
                "/predict",
                files={"file": (path.name, path.read_bytes(), "image/png")},
            )
            res.raise_for_status()
            body = res.json()
            ok = expected is None or body["digit"] == expected
            correct += int(ok)
            mark = "OK  " if ok else "MISS"
            print(f"{mark} {path.name:28s} -> {body['digit']} ({body['confidence']:.1%})")

    print(f"\n{correct}/{len(files)} correct ({correct / len(files):.1%})")


if __name__ == "__main__":
    main()
