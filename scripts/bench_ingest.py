"""Ingest throughput benchmark: serial vs full-core process pool.

Usage: python scripts/bench_ingest.py [n_files]
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from photomosaic.db import TileDB
from photomosaic.library import ingest


def generate(folder: str, n: int, size=(1200, 900), seed: int = 11) -> None:
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        arr[::7, :] = (i * 31) % 256          # structure so features differ
        Image.fromarray(arr).save(os.path.join(folder, f"img_{i:04d}.jpg"),
                                  "JPEG", quality=85)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    base = os.path.join(os.path.dirname(__file__), "..", "bench_ingest_data")
    base = os.path.abspath(base)
    folder = os.path.join(base, "photos")
    if not os.path.isdir(folder):
        print(f"generating {n} synthetic 1200x900 JPEGs ...")
        generate(folder, n)
    for workers, label in ((1, "serial "), (0, "all-core")):
        db_path = os.path.join(base, f"bench_w{workers}.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        with TileDB(db_path) as db:
            t0 = time.perf_counter()
            stats = ingest(db, folder, variants=("orig", "mirror"), workers=workers)
            dt = time.perf_counter() - t0
        print(f"{label} workers={workers or os.cpu_count():2}  "
              f"{dt:6.1f}s  {stats['files'] / dt:6.1f} files/s  rows={stats['rows']}")
    shutil.rmtree(base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
