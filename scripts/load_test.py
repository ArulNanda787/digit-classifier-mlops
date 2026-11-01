"""Generate traffic so the Grafana dashboard has something to show.

Sends mostly valid predictions plus a share of deliberately bad requests
(non-image, corrupt file, blank canvas) to exercise the error handling.

Usage:  python scripts/load_test.py --url http://<ec2-ip> --requests 300
"""

import argparse
import io
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _random_stroke() -> bytes:
    img = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(img)
    points = [(random.randint(60, 220), random.randint(40, 240)) for _ in range(4)]
    draw.line(points, fill=255, width=18)
    return _png(img)


def build_requests(n: int, error_share: float):
    samples = [p.read_bytes() for p in sorted(SAMPLES_DIR.glob("*.png"))]
    blank = _png(Image.new("L", (280, 280), 0))
    jobs = []
    for _ in range(n):
        if random.random() < error_share:
            jobs.append(random.choice([
                ("notes.txt", b"not an image", "text/plain"),       # -> 415
                ("broken.png", b"\x89PNG corrupted", "image/png"),  # -> 400
                ("blank.png", blank, "image/png"),                  # -> 400
            ]))
        else:
            data = random.choice(samples) if samples else _random_stroke()
            jobs.append(("digit.png", data, "image/png"))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="e.g. http://35.154.180.227")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--errors", type=float, default=0.15, help="share of bad requests")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between requests per worker")
    args = parser.parse_args()

    jobs = build_requests(args.requests, args.errors)
    statuses = Counter()
    latencies = []

    with httpx.Client(base_url=args.url, timeout=15) as client:
        def send(job):
            name, data, ctype = job
            start = time.perf_counter()
            try:
                res = client.post("/predict", files={"file": (name, data, ctype)})
                code = res.status_code
            except httpx.HTTPError as exc:
                code = type(exc).__name__
            latencies.append(time.perf_counter() - start)
            statuses[code] += 1
            time.sleep(args.delay)

        print(f"Sending {len(jobs)} requests to {args.url} with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(send, jobs))

    latencies.sort()
    print("\nStatus codes:", dict(statuses))
    print(f"Median latency: {latencies[len(latencies) // 2] * 1000:.0f} ms, "
          f"p95: {latencies[int(len(latencies) * 0.95)] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
