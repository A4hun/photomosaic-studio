"""Assignment planner: decides which tile goes into which leaf.

The classic approach matches every cell independently and picks one of the
K nearest tiles at random, which (a) repeats the same photo next to itself,
(b) over-uses a handful of "popular" colours and (c) ignores that a hard reuse
cap or a colour radius only helps *after* the damage is done.

This planner works in three steps:

1. **Candidates** - the GPU/CPU matcher returns, for every leaf, the K nearest
   tiles (streamed over tile blocks, so memory does not grow with the library).
2. **Fair-share ordering** - leaves are visited in a seeded random order, so no
   image region systematically gets the best tiles.
3. **Penalised choice** - each candidate is scored by its normalised colour
   error plus soft penalties: how often its *source photo* is already used
   (relative to its fair share) and whether the same photo already sits next to
   this leaf (checked on a coarse ownership map, across different leaf sizes).
   A little seeded noise keeps the "near-best random" variety knob (``top_k``).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

REUSE_WEIGHT = 0.5        # cost of using a photo `expected` times beyond fair share
ADJACENT_PENALTY = 4.0    # in units of the candidate pool's error span
JITTER = 0.8              # randomness amplitude (units of the pool's error span)


def plan_assignments(
    matcher,
    boxes: np.ndarray,
    feats: np.ndarray,
    avgs: np.ndarray,
    groups: np.ndarray,
    size,
    top_k: int = 8,
    seed: Optional[int] = None,
    max_reuse: int = 0,
    avoid_adjacent: bool = True,
    error_diffusion: bool = False,
    refine_passes: int = 0,
    progress: Optional[Callable[[str, int, int], None]] = None,
    check: Optional[Callable[[], None]] = None,
):
    """Return (tile positions (L,), distances (L,)) for every leaf.

    ``groups`` maps tile position -> source-photo id (variants of the same photo
    share one id).  ``check`` is called periodically and may raise to abort.

    error_diffusion=True 时按扫描序（从左到右、从上到下）处理，并把每个瓦片的
    平均色误差按 Floyd-Steinberg 权重扩散到右/下邻居——邻居选片时补偿，整张图
    远观收敛到主图，不再每格各自为政（业界成熟的拼图保真技术）。
    """
    L = int(boxes.shape[0])
    n = matcher.n
    top_k = max(1, int(top_k))
    strict = top_k <= 1            # K=1 keeps the classic "always the best match"
    if strict:
        avoid_adjacent = False
    k = min(n, max(16, min(64, 2 * top_k)))
    pool = min(top_k, k)

    cand_i, cand_d = matcher.candidates(
        feats, avgs, k,
        progress=(lambda d, t: progress("match", d, t)) if progress else None,
        check=check)

    rng = np.random.default_rng(seed)
    order = rng.permutation(L)
    jitter = (rng.random((L, k), dtype=np.float32) * JITTER) if top_k > 1 else None

    G = int(groups.max()) + 1 if groups.size else 1
    guse = np.zeros(G, dtype=np.float32)
    expected = max(1.0, L / float(max(1, G)))
    usage = matcher.usage                     # per tile position, shared with matcher

    picks = np.empty(L, dtype=np.int64)
    dists = np.empty(L, dtype=np.float64)

    # ---- coarse ownership map for the adjacency check ----------------------
    W, H = int(size[0]), int(size[1])
    if avoid_adjacent:
        min_side = max(1, int(min((boxes[:, 2] - boxes[:, 0]).min(),
                                  (boxes[:, 3] - boxes[:, 1]).min())))
        step = max(1, min_side // 2, int(np.ceil(max(W, H) / 3072.0)))
        oh, ow_ = -(-H // step), -(-W // step)
        owner = np.full((oh, ow_), -1, dtype=np.int32)
        ring_t = 2
        obox = np.empty((L, 4), dtype=np.int64)
        obox[:, 0] = boxes[:, 0] // step
        obox[:, 1] = boxes[:, 1] // step
        obox[:, 2] = np.maximum(obox[:, 0] + 1, -(-boxes[:, 2] // step))
        obox[:, 3] = np.maximum(obox[:, 1] + 1, -(-boxes[:, 3] // step))
        np.clip(obox[:, [0, 2]], 0, ow_, out=obox[:, [0, 2]])
        np.clip(obox[:, [1, 3]], 0, oh, out=obox[:, [1, 3]])

    # ---- 误差扩散：叶子 → 粗网格行列，扫描序处理 + Floyd-Steinberg 误差扩散 ----
    err = None
    grid_rc = None
    tile_avg = None
    if error_diffusion or refine_passes > 0:
        # 瓦片平均色（转成 numpy；GPU 时在 CPU 上取一份）
        tile_avg = np.asarray(matcher.avgs if not hasattr(matcher.avgs, "get")
                              else matcher.avgs.get(), dtype=np.float64)
    if error_diffusion:
        # 叶子按其中心映射到粗网格（自适应大小格混排也按中心归位）
        min_w = max(1, int((boxes[:, 2] - boxes[:, 0]).min()))
        min_h = max(1, int((boxes[:, 3] - boxes[:, 1]).min()))
        cols = max(1, int(np.ceil(W / min_w)))
        rows = max(1, int(np.ceil(H / min_h)))
        cx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2 / W * cols).astype(np.int64), 0, cols - 1)
        cy = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2 / H * rows).astype(np.int64), 0, rows - 1)
        grid_rc = cy * cols + cx
        err = np.zeros((rows, cols, 3), dtype=np.float64)
        # 扫描序：按 (行, 列) 排序（误差扩散必须是因果顺序）
        order = np.argsort(grid_rc, kind="stable")

    for t, i in enumerate(order):
        if t % 512 == 0:
            if check:
                check()
            if progress and t % 2048 == 0:
                progress("plan", t, L)
        ci = cand_i[i]
        cd = cand_d[i]
        cg = groups[ci]
        d0 = float(cd[0])
        chosen = -1
        ring = None
        if avoid_adjacent:
            x0, y0, x1, y1 = obox[i]
            parts = []
            ya, yb = max(0, y0 - ring_t), min(oh, y1 + ring_t)
            xa, xb = max(0, x0 - ring_t), min(ow_, x1 + ring_t)
            if ya < y0:
                parts.append(owner[ya:y0, xa:xb].ravel())
            if y1 < yb:
                parts.append(owner[y1:yb, xa:xb].ravel())
            if xa < x0:
                parts.append(owner[y0:y1, xa:x0].ravel())
            if x1 < xb:
                parts.append(owner[y0:y1, x1:xb].ravel())
            if parts:
                ring = np.unique(np.concatenate(parts))
                ring = ring[ring >= 0]
                if ring.size == 0:
                    ring = None
        if np.isfinite(d0):
            span = float(cd[pool - 1]) - d0
            if not np.isfinite(span) or span <= 1e-6:
                span = max(abs(d0) * 0.25, 1.0)
            cost = (cd - d0) / span
            if not strict:
                cost += (REUSE_WEIGHT / expected) * guse[cg]
            if ring is not None:
                if ring.size <= 48:
                    hit = (cg[:, None] == ring[None, :]).any(axis=1)
                else:
                    hit = np.isin(cg, ring)
                cost += ADJACENT_PENALTY * hit
            if max_reuse > 0:
                cost[usage[ci] >= max_reuse] = np.inf
            cost[~np.isfinite(cd)] = np.inf
            if jitter is not None:
                cost += jitter[i]
            # 误差扩散：偏向能补偿累计色偏的候选（远观收敛到主图）
            cy = cx = None
            target = None
            if error_diffusion:
                cy, cx = divmod(int(grid_rc[i]), err.shape[1])
                target = avgs[i] + err[cy, cx]
                rgb_err = ((tile_avg[ci] - target) ** 2).sum(axis=1)
                span_rgb = max(1.0, float(np.ptp(tile_avg[ci], axis=0).sum()))
                cost += 2.0 * rgb_err / span_rgb          # 与特征误差同量级
            j = int(np.argmin(cost))
            if np.isfinite(cost[j]):
                chosen = int(ci[j])
                picks[i] = chosen
                dists[i] = float(cd[j])
                usage[chosen] += 1
                if error_diffusion and target is not None:
                    e = target - tile_avg[chosen]         # 本格的色偏
                    H_, W_ = err.shape[0], err.shape[1]
                    if cx + 1 < W_:
                        err[cy, cx + 1] += e * (7 / 16)
                    if cy + 1 < H_:
                        if cx > 0:
                            err[cy + 1, cx - 1] += e * (3 / 16)
                        err[cy + 1, cx] += e * (5 / 16)
                        if cx + 1 < W_:
                            err[cy + 1, cx + 1] += e * (1 / 16)
        if chosen < 0:
            # nothing usable among the candidates (colour radius / reuse cap):
            # exact, cap-aware pick over the whole library.
            p, dd = matcher.select_batch(feats[i:i + 1], avgs[i:i + 1])
            chosen = int(p[0])
            picks[i] = chosen
            dists[i] = float(dd[0])
        g = int(groups[chosen])
        guse[g] += 1
        if avoid_adjacent:
            x0, y0, x1, y1 = obox[i]
            owner[y0:y1, x0:x1] = g
    # ---- 迭代精炼（原版 Artensoft 的 "Improving Quality"）------------------
    # 贴完后再改：每轮取色偏最差的格子，在候选池里换上平均色更贴的瓦片。
    # 特征距离保证结构可接受（候选本来就是 top-k），换的时候只优化颜色。
    for _pass in range(max(0, int(refine_passes))):
        if check:
            check()
        if progress:
            progress("refine", _pass, int(refine_passes))
        cur_err = ((tile_avg[picks] - avgs) ** 2).sum(axis=1)      # (L,) 色偏²
        if not np.isfinite(cur_err).all() or cur_err.sum() <= 1e-9:
            break
        worst = np.argsort(cur_err)[::-1][: max(1, L // 8)]        # 最差 12.5%
        improved = 0
        for i in worst:
            ci = cand_i[i]
            cd = cand_d[i]
            ok = np.isfinite(cd)
            if not ok.any():
                continue
            # 候选的色偏²（目标 = 本格主图均色）
            errs = ((tile_avg[ci] - avgs[i]) ** 2).sum(axis=1)
            errs[~ok] = np.inf
            if max_reuse > 0:
                errs[usage[ci] >= max_reuse] = np.inf
            j = int(np.argmin(errs))
            if np.isfinite(errs[j]) and int(ci[j]) != int(picks[i]) \
                    and errs[j] < cur_err[i]:
                old = int(picks[i])
                new = int(ci[j])
                usage[old] -= 1
                guse[int(groups[old])] -= 1
                usage[new] += 1
                guse[int(groups[new])] += 1
                picks[i] = new
                dists[i] = float(cd[j])
                cur_err[i] = errs[j]
                improved += 1
        if improved == 0:
            break
    if progress:
        progress("plan", L, L)
    return picks, dists
