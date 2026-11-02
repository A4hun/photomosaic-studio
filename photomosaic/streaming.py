"""分条带流式渲染：百亿像素级输出的内存安全路径。

整图路径画布常驻内存（每 10 亿像素 ≈ 3 GB），超大输出必然 OOM。本模块按
水平条带渲染：每个条带只分配一条的画布，把与条带相交的叶子瓦片 resize 到
叶子完整输出尺寸后裁剪相交部分贴入（与整图路径逐像素一致），写进
BigTIFF 后立即释放。条带偏移在打开文件时就固定，崩溃/取消后可原位续写。

与整图路径逐像素一致的关键：每张瓦片的全局解码尺寸（所有叶子中的最大
尺寸）在条带循环之前一次算好，条带内共用——这样 JPEG draft 的解码缩放
级别与整图路径完全相同。条带内的贴图任务再按字节预算分块解码（大库防
OOM），分块不改变解码尺寸与贴图结果。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .bigtiff import BigTiffStripWriter, estimate_size, strip_layout
from .db import TileDB

# 单条带目标内存预算（字节，RGB）
_STRIP_BUDGET = 384 * 1024 * 1024
# 条带内贴图分块预算：估算字节（Σ 单叶子 w*h*3）超过或 job 数超过上限就
# 切一块。分块只影响"何时取库/解码"，不影响解码尺寸与贴图内容。
_CHUNK_BUDGET_BYTES = 128 * 1024 * 1024
_CHUNK_BUDGET_JOBS = 2048
# decoded 缓存的估算字节上限：块结束时超过就整表清空（与 render._paste_tiles
# 的 600MB 上限一致）——大库单条带数千张去重瓦片不会同时驻留。
_DECODE_CAP_BYTES = 600 * 1024 * 1024
# 降采样预览的最大宽度
_PREVIEW_MAX_W = 4096


def choose_strip_rows(width: int, height: int) -> int:
    """按内存预算挑条带高度：整图就一条带时也不小于 1 行。"""
    width, height = int(width), int(height)
    if width < 1:
        return 1
    rows = max(1, _STRIP_BUDGET // (width * 3))
    return min(rows, height)


def estimate_streaming_size(width: int, height: int, rows_per_strip: int) -> int:
    """流式 BigTIFF 输出的预计字节数（渲染前磁盘空间检查用）。"""
    return estimate_size(width, height, rows_per_strip)


class _PreviewAccumulator:
    """把每个条带降采样后纵向拼成一张预览图（不落整图内存）。"""

    def __init__(self, width: int, height: int):
        self.ow = min(width, _PREVIEW_MAX_W)
        self.scale = self.ow / max(1, width)
        self.oh = max(1, int(round(height * self.scale)))
        self._parts: List[np.ndarray] = []

    def add_strip(self, arr: np.ndarray, rows: int) -> None:
        target = max(1, int(round(rows * self.scale)))
        img = Image.fromarray(arr).resize((self.ow, target), Image.Resampling.BOX)
        self._parts.append(np.asarray(img, dtype=np.uint8))

    def finish(self, path: str) -> str:
        rows = [p for p in self._parts if p.shape[0] > 0]
        if not rows:
            return ""
        out = np.vstack(rows)
        if out.shape[0] != self.oh:      # 舍入误差 → 重采样对齐
            out = np.asarray(Image.fromarray(out).resize(
                (self.ow, self.oh), Image.Resampling.BOX))
        Image.fromarray(out).save(path)
        return path


def render_streaming(
    leaves,                       # [(x0,y0,x1,y1,level), ...] 主图坐标
    picks,                        # 每叶瓦片索引
    ids, metas,                   # 库数组
    db: TileDB,
    out_path: str,
    OW: int, OH: int, s: float,
    blend_pct: int,
    avgs: Optional[np.ndarray] = None,
    preview_path: Optional[str] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    status: Optional[Callable[[str], None]] = None,
    done_strips: int = 0,
    ckpt_strips: Optional[Callable[[int], None]] = None,
    strip_rows: Optional[int] = None,
) -> Dict[str, object]:
    """条带式渲染 `leaves` 到 BigTIFF；返回统计（与整图路径逐像素一致）。"""
    from .render import RenderCancelled, _decode_tile, _fit_tile

    total = len(leaves)
    if strip_rows:
        rps, n_strips = strip_layout(OW, OH, strip_rows)
    else:
        rps, n_strips = strip_layout(OW, OH, choose_strip_rows(OW, OH))
    if status:
        status(f"正在流式渲染（{n_strips} 条带 × {rps} 行）…")
    preview = _PreviewAccumulator(OW, OH) if preview_path else None

    # 每叶输出坐标 + 选中瓦片 + 层级直方图（一次算好）
    leaf_rects: List[Tuple[int, int, int, int, int]] = []
    chosen: List[int] = []
    histogram: Dict[int, int] = {}
    for i, (x0, y0, x1, y1, level) in enumerate(leaves):
        ox0, oy0 = int(round(x0 * s)), int(round(y0 * s))
        ox1, oy1 = int(round(x1 * s)), int(round(y1 * s))
        histogram[int(level)] = histogram.get(int(level), 0) + 1
        leaf_rects.append((ox0, oy0, ox1, oy1, int(level)))
        chosen.append(int(ids[int(picks[i])]))

    # 全局解码尺寸表：与整图路径 _paste_tiles 的 need 完全一致（保证 draft
    # 解码缩放级别相同 → 逐像素一致）。
    global_need: Dict[int, Tuple[int, int]] = {}
    leaf_size: List[Tuple[int, int]] = []
    leaf_tint: List[Optional[Tuple[int, int, int]]] = []
    for i, (ox0, oy0, ox1, oy1, _lvl) in enumerate(leaf_rects):
        size = (max(1, ox1 - ox0), max(1, oy1 - oy0))
        leaf_size.append(size)
        if avgs is not None:
            leaf_tint.append((int(avgs[i][0]), int(avgs[i][1]), int(avgs[i][2])))
        else:
            leaf_tint.append(None)
        tid = chosen[i]
        o = global_need.get(tid)
        global_need[tid] = (max(o[0], size[0]), max(o[1], size[1])) if o else size

    # 条带候选：按 y0 排序后，条带 [sy0, sy1) 的候选 = 所有 y0 < sy1 的叶子
    # 中 y1 > sy0 的那些（长叶子会跨多条带，每条都要贴相交部分）。
    ys = np.asarray([r[1] for r in leaf_rects], dtype=np.int64)
    ye = np.asarray([r[3] for r in leaf_rects], dtype=np.int64)
    order = np.argsort(ys, kind="stable")
    ys_sorted = ys[order]
    ye_sorted = ye[order]

    def paste_strip(strip: Image.Image, sy0: int, sy1: int) -> None:
        lo = int(np.searchsorted(ys_sorted, sy1, side="right"))
        if lo <= 0:
            return
        idx = order[:lo]
        mask = ye_sorted[:lo] > sy0
        cand = idx[mask]
        if cand.size == 0:
            return
        jobs = []
        for i in cand.tolist():
            i = int(i)
            ox0, oy0, ox1, oy1, _lvl = leaf_rects[i]
            if oy1 <= sy0 or oy0 >= sy1:
                continue
            jobs.append((i, ox0, oy0, leaf_size[i], chosen[i], leaf_tint[i]))
        if not jobs:
            return
        # ---- 分块解码（复刻 render._paste_tiles 的模式）--------------------
        # 条带内 jobs 按"估算字节 = Σ 单叶子 w*h*3"分块：每块批量取库、线程
        # 池并行解码，避免大库（单条带数千张去重瓦片）一次全部解码驻留导致
        # OOM。叶子互不相交，贴图顺序无关；关键约束：每张瓦片无论落在哪个
        # 块，解码尺寸都必须是 global_need[tid]（全局最大，绝不能改成块内
        # 最大），这正是与整图路径逐像素一致的机制。
        chunks: List[List[tuple]] = []
        cur: List[tuple] = []
        cur_est = 0
        for j in jobs:
            w, h = j[3]
            cur.append(j)
            cur_est += w * h * 3
            if cur_est >= _CHUNK_BUDGET_BYTES or len(cur) >= _CHUNK_BUDGET_JOBS:
                chunks.append(cur)
                cur, cur_est = [], 0
        if cur:
            chunks.append(cur)

        decoded: Dict[int, Image.Image] = {}
        decoded_est = 0                    # decoded 内图像的估算字节数
        workers = max(2, min(16, os.cpu_count() or 4))
        with ThreadPoolExecutor(workers) as ex:
            for chunk in chunks:
                todo = [tid for tid in sorted({j[4] for j in chunk})
                        if tid not in decoded]
                if todo:
                    raws = db.get_thumbs(todo)
                    srcs = db.get_sources(todo)
                    for tid, img in zip(todo, ex.map(
                            lambda tid: _decode_tile(raws[tid], global_need[tid],
                                                     srcs.get(tid)), todo)):
                        decoded[tid] = img
                        nw, nh = global_need[tid]
                        decoded_est += nw * nh * 3
                    del raws
                for i, ox0, oy0, size, tid, tint in chunk:
                    img = _fit_tile(decoded[tid], size, blend_pct, tint)
                    iy0 = max(oy0, sy0)             # 与条带的相交区（整图坐标）
                    iy1 = min(oy0 + size[1], sy1)
                    if iy1 <= iy0:
                        continue
                    if iy0 != oy0 or iy1 != oy0 + size[1]:
                        img = img.crop((0, iy0 - oy0, size[0], iy1 - oy0))
                    strip.paste(img, (ox0, iy0 - sy0))
                if decoded_est > _DECODE_CAP_BYTES:   # 与 _paste_tiles 的上限一致
                    decoded.clear()
                    decoded_est = 0

    w = BigTiffStripWriter(out_path, OW, OH, rps, append=done_strips > 0)
    try:
        for si in range(done_strips, n_strips):
            if should_cancel and should_cancel():
                if ckpt_strips:
                    ckpt_strips(si)
                raise RenderCancelled("cancelled during streaming paste")
            sy0 = si * rps
            sy1 = min(OH, sy0 + rps)
            strip = Image.new("RGB", (OW, sy1 - sy0), (255, 255, 255))
            paste_strip(strip, sy0, sy1)
            w.write_strip(si, np.asarray(strip, dtype=np.uint8))
            w.flush(sync=(si + 1 == n_strips))
            if preview is not None:
                preview.add_strip(np.asarray(strip), sy1 - sy0)
            del strip
            if progress:
                done = si + 1 - done_strips
                progress(done, max(1, n_strips - done_strips))
            if ckpt_strips:
                ckpt_strips(si + 1)
    finally:
        w.close()

    prev_path = preview.finish(preview_path) if preview is not None else ""
    return {
        "strips": n_strips,
        "strip_rows": rps,
        "streamed": True,
        "bytes": os.path.getsize(out_path),
        "preview": prev_path,
        "leaf_rects": leaf_rects,
        "choices": chosen,
        "level_histogram": histogram,
    }
