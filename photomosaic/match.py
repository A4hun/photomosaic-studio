"""Tile matching with an optional GPU backend.

Distance between a master cell and a tile is the squared Euclidean distance
of their 27-dimensional 3x3-LAB feature vectors (no square root: monotone and
cheaper).  Selection is *near-best random*: the top-K closest candidates are
shortlisted and one is drawn uniformly at random, which keeps the result
visually faithful while avoiding the repetitive look of always taking the
single best match.

Performance: distances for a whole chunk of cells are computed with one
matrix product (BLAS on CPU, cuBLAS on GPU via CuPy when available), and the
top-K shortlist is obtained with a vectorised argpartition.  An optional
average-RGB radius prefilter and a per-tile reuse cap mirror the classic
controls; the cap is enforced exactly with a small sequential fix-up pass.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


_BACKENDS: Dict[str, Tuple] = {}


def resolve_backend(device: str = "auto") -> Tuple[object, str, object]:
    """Return (array module, device name, to-numpy converter), cached per process."""
    if device in _BACKENDS:
        return _BACKENDS[device]
    if device in ("auto", "gpu"):
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                try:        # fail fast (and let callers shrink blocks) instead of
                    cp.get_default_memory_pool().set_limit(fraction=0.85)  # thrashing
                except Exception:
                    pass
                _BACKENDS[device] = (cp, "gpu", cp.asnumpy)
                return _BACKENDS[device]
        except Exception:
            if device == "gpu":
                raise RuntimeError(
                    "未检测到 CuPy（GPU 匹配后端）。"
                    "请改用 --device cpu，或安装 cupy-cuda12x 等与本机 CUDA 匹配的包。"
                ) from None
    _BACKENDS[device] = (np, "cpu", lambda a: a)
    return _BACKENDS[device]


class Matcher:
    def __init__(
        self,
        tiles: Dict[str, object],
        top_k: int = 8,
        radius: Optional[float] = None,
        max_reuse: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        chunk: Optional[int] = None,
    ):
        self.xp, self.device, self._asnp = resolve_backend(device)
        xp = self.xp
        self.feats = xp.asarray(tiles["feats"])          # (N, 27)
        self.avgs = xp.asarray(tiles["avgs"])            # (N, 3)
        self.n = int(self.feats.shape[0])
        if self.n == 0:
            raise ValueError("tile library is empty — ingest a folder first")
        self.top_k = max(1, int(top_k))
        self.radius = float(radius) if radius and radius > 0 else None
        self.max_reuse = int(max_reuse)                  # 0 = unlimited
        self.usage = np.zeros(self.n, dtype=np.int32)    # stays on CPU
        self.rng = np.random.default_rng(seed)
        self.chunk = int(chunk) if chunk else self._auto_chunk()
        # |f|^2 precomputed; d(c) = |f|^2 - 2 f·c + |c|^2
        self._feat_norms = xp.einsum("ij,ij->i", self.feats, self.feats)
        self._feats_T = self.feats.T

    def _auto_chunk(self) -> int:
        """Rows per distance block.

        A block materialises a (C, N) float32 matrix plus argpartition's int64
        index copies (~5x the matrix), so C is sized from N and, on GPU, from
        the memory that is actually free -- a fixed C overflows large libraries.
        """
        if self.device != "gpu":
            return 512
        try:
            free, _total = self.xp.cuda.runtime.memGetInfo()
        except Exception:
            free = 4 << 30
        per_row = max(1, self.n) * 40            # bytes: D + sort/index temporaries
        return int(max(64, min(4096, (free * 0.45) // per_row)))

    def _release_gpu(self) -> None:
        if self.device == "gpu":
            try:
                self.xp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    # ---- distances -------------------------------------------------------
    def _dist_chunk(self, cell_feats: np.ndarray) -> object:
        """(C,27) cpu array -> (C,N) distance matrix on the active device."""
        xp = self.xp
        F = xp.asarray(cell_feats, dtype=xp.float32)
        D = self._feat_norms[None, :] - 2.0 * (F @ self._feats_T)
        D += xp.einsum("ij,ij->i", F, F)[:, None]
        xp.maximum(D, 0.0, out=D)
        return D

    def _apply_filters(self, D, cell_avgs: np.ndarray) -> None:
        xp = self.xp
        if self.radius is not None:
            A = xp.asarray(cell_avgs, dtype=xp.float32)
            # |a-c|^2 = |a|^2 - 2 a.c + |c|^2  (no (C,N,3) intermediate)
            an = xp.einsum("ij,ij->i", self.avgs, self.avgs)
            cn = xp.einsum("ij,ij->i", A, A)
            da = an[None, :] - 2.0 * (A @ self.avgs.T) + cn[:, None]
            D[da > self.radius * self.radius] = xp.inf
        if self.max_reuse > 0:
            over = xp.asarray(self.usage >= self.max_reuse)
            if bool(over.any()):
                D[:, over] = xp.inf

    def _topk_idx(self, D, k: int):
        """(C,N) -> (C,k) column indices of a top-k shortlist per row.

        CPU: row-parallel argpartition (numpy releases the GIL in partition).
        """
        xp = self.xp
        C = D.shape[0]
        if xp is not np or C < 128 or self.n < 4096:
            return xp.argpartition(D, k - 1, axis=1)[:, :k]
        import concurrent.futures as cf
        import os

        nthreads = max(1, min(8, os.cpu_count() or 1, C // 128))
        bounds = np.linspace(0, C, nthreads + 1).astype(int)
        out = np.empty((C, k), dtype=np.int64)

        def work(i: int) -> None:
            a, b = int(bounds[i]), int(bounds[i + 1])
            if b > a:
                out[a:b] = np.argpartition(D[a:b], k - 1, axis=1)[:, :k]

        with cf.ThreadPoolExecutor(nthreads) as ex:
            list(ex.map(work, range(nthreads)))
        return out

    def _pick_rows(self, D) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised near-best random pick per row, then exact fix-up for
        rows whose draw is masked (radius) or violates the reuse cap."""
        xp = self.xp
        C = D.shape[0]
        k = min(self.top_k, self.n)
        part = self._topk_idx(D, k)                            # (C,k) tile ids
        rows = xp.arange(C)[:, None]
        cols = xp.asarray(self.rng.integers(0, k, size=C))[:, None]
        pick_xp = part[rows, cols]                             # (C,1) tile ids
        picks = self._asnp(pick_xp.ravel())
        dists = self._asnp(D[rows, pick_xp].ravel())
        for i in range(C):
            p = picks[i]
            capped = self.max_reuse > 0 and self.usage[p] >= self.max_reuse
            if not np.isfinite(dists[i]) or capped:
                drow = self._asnp(D[i])
                order = np.argsort(drow, kind="stable")
                chosen = order[0]
                for cand in order:
                    ok_finite = np.isfinite(drow[cand])
                    ok_cap = self.max_reuse == 0 or self.usage[cand] < self.max_reuse
                    if ok_finite and ok_cap:
                        chosen = cand
                        break
                picks[i], dists[i] = int(chosen), float(drow[chosen])
            self.usage[picks[i]] += 1
        return picks, dists

    # ---- streaming top-K candidates ----------------------------------------
    def release(self) -> None:
        """Hand cached GPU blocks back (call when a render finishes)."""
        self._release_gpu()

    def _cand_rows(self, bt: int) -> int:
        if self.device != "gpu":
            return 256
        try:
            free, _total = self.xp.cuda.runtime.memGetInfo()
        except Exception:
            free = 4 << 30
        # per row: D (4 B) + argpartition index/temporaries (~24 B) per tile of a block
        return int(max(32, min(2048, (free * 0.35) // (bt * 28))))

    def _cand_block(self, cell_feats: np.ndarray, cell_avgs: np.ndarray, k: int, bt: int):
        """Top-k tiles for a block of cells, streamed over tile blocks so peak
        memory is O(rows * bt) regardless of library size."""
        xp = self.xp
        F = xp.asarray(cell_feats, dtype=xp.float32)
        fn = xp.einsum("ij,ij->i", F, F)[:, None]
        A = an = cn = None
        if self.radius is not None:
            A = xp.asarray(cell_avgs, dtype=xp.float32)
            cn = xp.einsum("ij,ij->i", A, A)[:, None]
            an = xp.einsum("ij,ij->i", self.avgs, self.avgs)
        r2 = (self.radius or 0.0) ** 2
        best_d = best_i = None
        for t0 in range(0, self.n, bt):
            t1 = min(self.n, t0 + bt)
            D = self._feat_norms[None, t0:t1] - 2.0 * (F @ self._feats_T[:, t0:t1])
            D += fn
            xp.maximum(D, 0.0, out=D)
            if A is not None:
                da = an[None, t0:t1] - 2.0 * (A @ self.avgs[t0:t1].T) + cn
                D[da > r2] = xp.inf
                del da
            kk = min(k, t1 - t0)
            part = self._topk_idx(D, kk)
            idx = xp.ascontiguousarray(part[:, :kk])
            del part
            d = xp.take_along_axis(D, idx, axis=1)
            del D
            idx = idx + t0
            if best_d is None:
                best_d, best_i = d, idx
            else:
                bd = xp.concatenate([best_d, d], axis=1)
                bi = xp.concatenate([best_i, idx], axis=1)
                if bd.shape[1] > k:
                    sel = xp.argpartition(bd, k - 1, axis=1)[:, :k]
                    bd = xp.take_along_axis(bd, sel, axis=1)
                    bi = xp.take_along_axis(bi, sel, axis=1)
                best_d, best_i = bd, bi
        order = xp.argsort(best_d, axis=1)
        return xp.take_along_axis(best_i, order, axis=1), xp.take_along_axis(best_d, order, axis=1)

    def candidates(self, cell_feats: np.ndarray, cell_avgs: np.ndarray, k: int,
                   progress=None, check=None) -> Tuple[np.ndarray, np.ndarray]:
        """(L,k) nearest tile positions and squared distances, ascending.

        The radius prefilter is applied (excluded tiles come back with an
        infinite distance); reuse caps are left to the caller because they
        change while assignments are made.
        """
        cell_feats = np.asarray(cell_feats, dtype=np.float32)
        cell_avgs = np.asarray(cell_avgs, dtype=np.float32)
        L = cell_feats.shape[0]
        k = max(1, min(int(k), self.n))
        bt = min(self.n, 65536 if self.device == "gpu" else 32768)
        rows = self._cand_rows(bt)
        out_i = np.empty((L, k), dtype=np.int64)
        out_d = np.empty((L, k), dtype=np.float32)
        s = 0
        while s < L:
            if check:
                check()
            e = min(L, s + rows)
            try:
                bi, bd = self._cand_block(cell_feats[s:e], cell_avgs[s:e], k, bt)
                out_i[s:e] = self._asnp(bi)
                out_d[s:e] = self._asnp(bd)
            except Exception as exc:
                if self.device == "gpu" and "OutOfMemory" in type(exc).__name__ \
                        and rows > 16:
                    self._release_gpu()
                    rows = max(16, rows // 2)
                    continue
                raise
            s = e
            if progress:
                progress(s, L)
        return out_i, out_d

    # ---- public API --------------------------------------------------------
    def select_batch(
        self, cell_feats: np.ndarray, cell_avgs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match many cells at once. Returns (tile positions, distances)."""
        cell_feats = np.asarray(cell_feats, dtype=np.float32)
        cell_avgs = np.asarray(cell_avgs, dtype=np.float32)
        picks: List[int] = []
        dists: List[float] = []
        s = 0
        total = cell_feats.shape[0]
        while s < total:
            e = min(total, s + self.chunk)
            try:
                D = self._dist_chunk(cell_feats[s:e])
                self._apply_filters(D, cell_avgs[s:e])
                p, d = self._pick_rows(D)
            except Exception as exc:               # GPU out of memory -> smaller blocks
                if self.device == "gpu" and "OutOfMemory" in type(exc).__name__ \
                        and self.chunk > 32:
                    D = None
                    self._release_gpu()
                    self.chunk = max(32, self.chunk // 2)
                    continue                        # retry the same slice
                raise
            picks.extend(p.tolist())
            dists.extend(d.tolist())
            del D
            s = e
        return np.asarray(picks, dtype=np.int64), np.asarray(dists, dtype=np.float64)

    def select(self, cell_feat: np.ndarray, cell_avg: np.ndarray) -> Tuple[int, float]:
        p, d = self.select_batch(cell_feat[None, :], cell_avg[None, :])
        return int(p[0]), float(d[0])

    def top_candidates(self, cell_feat: np.ndarray, k: int = 12) -> List[Tuple[int, float]]:
        """The k nearest tile positions as (position, distance), sorted."""
        D = self._dist_chunk(np.asarray(cell_feat, dtype=np.float32)[None, :])
        d = self._asnp(D[0])
        k = max(1, min(int(k), self.n))
        order = np.argpartition(d, k - 1)[:k]
        order = order[np.argsort(d[order], kind="stable")]
        return [(int(i), float(d[i])) for i in order]
