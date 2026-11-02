"""Self-contained demo: generates a synthetic tile library and a master
image, renders a mosaic, and prints statistics.

Usage:  python scripts/demo.py [output_dir]
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.render import make_mosaic


def generate_tiles(folder: str, count: int = 80, size: int = 128, seed: int = 7) -> None:
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(count):
        base = rng.integers(30, 226, 3)
        arr = np.empty((size, size, 3), dtype=np.uint8)
        yy, xx = np.mgrid[0:size, 0:size]
        style = i % 4
        if style == 0:  # gradient around a random hue
            arr[..., 0] = np.clip(base[0] + xx * 0.8, 0, 255)
            arr[..., 1] = np.clip(base[1] + yy * 0.5, 0, 255)
            arr[..., 2] = base[2]
        elif style == 1:  # blobs
            arr[:] = base
            for _ in range(4):
                cx, cy = rng.integers(0, size, 2)
                r = rng.integers(10, size // 2)
                colour = rng.integers(0, 256, 3)
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
                arr[mask] = colour
        elif style == 2:  # stripes
            freq = rng.integers(4, 16)
            stripe = ((xx // freq) % 2 == 0)[..., None]
            arr[:] = np.where(stripe, base, rng.integers(0, 256, 3))
        else:  # noise-ish texture
            arr[:] = np.clip(
                base + rng.normal(0, 40, (size, size, 3)), 0, 255
            ).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(folder, f"tile_{i:03d}.png"))


def generate_master(path: str, size: int = 640) -> None:
    img = Image.new("RGB", (size, size), (245, 245, 250))
    d = ImageDraw.Draw(img)
    # sky-sunset gradient background
    for y in range(size):
        t = y / size
        d.line([(0, y), (size, y)],
               fill=(int(30 + 200 * t), int(120 - 60 * t), int(220 - 140 * t)))
    # sun
    d.ellipse([size * 0.62, size * 0.12, size * 0.86, size * 0.36], fill=(255, 220, 80))
    # mountains
    d.polygon([(0, size * 0.72), (size * 0.3, size * 0.38), (size * 0.58, size * 0.72)],
              fill=(60, 70, 110))
    d.polygon([(size * 0.35, size * 0.72), (size * 0.68, size * 0.3), (size, size * 0.72)],
              fill=(40, 48, 84))
    # water
    d.rectangle([0, size * 0.72, size, size], fill=(24, 60, 110))
    img.save(path)


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
            os.path.dirname(__file__), "..", "demo_output")
    out_dir = os.path.abspath(out_dir)
    tiles_dir = os.path.join(out_dir, "tiles")
    os.makedirs(out_dir, exist_ok=True)

    master_path = os.path.join(out_dir, "master.png")
    db_path = os.path.join(out_dir, "tiles.db")
    mosaic_path = os.path.join(out_dir, "mosaic.png")

    print(f"generating demo data in {out_dir} ...")
    generate_tiles(tiles_dir)
    generate_master(master_path)

    if os.path.exists(db_path):
        os.remove(db_path)
    with TileDB(db_path) as db:
        stats = ingest(db, tiles_dir, ratio=1.0,
                       variants=("orig", "mirror", "r90", "r180", "r270"))
        print(f"ingest: {stats}")
        print(f"tiles in db: {db.count()}")
        result = make_mosaic(
            master_path, db, mosaic_path,
            cells_x=40, cell_px=48, top_k=6, max_reuse=40, seed=2026,
        )
    print("mosaic stats (uniform):")
    for k, v in result.items():
        print(f"  {k}: {v}")

    with TileDB(db_path) as db:
        adaptive_path = os.path.join(out_dir, "mosaic_adaptive.png")
        # manual region: force fine tiles over the sun area (like the
        # original tool's "set this area to more detail" feature)
        result2 = make_mosaic(
            master_path, db, adaptive_path,
            cells_x=10, levels=3, detail_threshold=4.0,
            regions=[(384, 64, 576, 256, 3)],
            top_k=6, max_reuse=60, seed=2026,
        )
    print("mosaic stats (adaptive multi-level):")
    for k, v in result2.items():
        print(f"  {k}: {v}")
    print(f"\nopen: {mosaic_path}\nopen: {adaptive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
