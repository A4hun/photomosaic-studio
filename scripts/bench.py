"""Matching backend benchmark: L cells x N tiles, CPU (BLAS) vs GPU (cuBLAS).

Usage: python scripts/bench.py [N] [L]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from photomosaic.match import Matcher  # noqa: E402


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    rng = np.random.default_rng(0)
    tiles = {
        "ids": np.arange(N, dtype=np.int64),
        "feats": rng.normal(50, 10, (N, 27)).astype(np.float32),
        "avgs": rng.uniform(0, 255, (N, 3)).astype(np.float32),
        "metas": [{"id": i} for i in range(N)],
    }
    feats = rng.normal(50, 10, (L, 27)).astype(np.float32)
    avgs = rng.uniform(0, 255, (L, 3)).astype(np.float32)
    print(f"benchmark: {L} cells x {N} tiles, top_k=8")
    for dev in ("cpu", "gpu"):
        try:
            t0 = time.perf_counter()
            m = Matcher(tiles, top_k=8, seed=1, device=dev)
            picks, dists = m.select_batch(feats, avgs)
            dt = time.perf_counter() - t0
            print(f"  {dev:4} -> device={m.device:3} {dt:7.2f}s "
                  f"({L / dt:,.0f} cells/s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  {dev:4} -> unavailable: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
