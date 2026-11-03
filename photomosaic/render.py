"""Master image grid analysis and final mosaic assembly.

Rendering modes:

* uniform  – regular cells_x x cells_y grid, fixed output tile size;
* adaptive – quadtree over the base grid driven by local detail energy and/or
  manual forced regions (the multi-level "staircase");
* manual   – an explicit leaf list (e.g. from the interactive grid editor),
  optionally refined further by the adaptive pass.

Performance notes: per-rect 3x3-LAB features come from half/quarter-resolution
integral images (O(1) per rect, fully vectorised over all cells), and tile
matching runs as chunked matrix products on CPU (BLAS) or GPU (CuPy/cuBLAS).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from PIL import ImageChops

from .db import TileDB
from .features import average_rgb, crop_to_ratio, lab_array, subblock_lab
from .library import VARIANTS
from .match import Matcher
from .planner import plan_assignments

# Print-size output is an intended use: lift Pillow's decompression-bomb pixel cap
# (make_mosaic enforces its own, explicit output-size limit).
Image.MAX_IMAGE_PIXELS = None

# 超过该输出像素数 → 切换分条带流式渲染（BigTIFF）；GUI 阈值提示与 CLI 默认共用
STREAM_THRESHOLD_PX = 200_000_000

Box = Tuple[int, int, int, int]
BaseBox = Tuple[Box, int]  # (rect, starting level)


class RenderCancelled(Exception):
    """Raised when should_cancel() fires; a checkpoint is written first."""


def master_cell_features(
    master: Image.Image, cells_x: int, cells_y: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell 27-dim LAB features and average RGB for a uniform grid."""
    rgb = master.convert("RGB")
    avg_grid = np.asarray(
        rgb.resize((cells_x, cells_y), Image.Resampling.BOX), dtype=np.float32
    )  # (cy, cx, 3)
    big = rgb.resize((cells_x * 3, cells_y * 3), Image.Resampling.BOX)
    arr = lab_array(big)
    feats = (
        arr.reshape(cells_y, 3, cells_x, 3, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(cells_y, cells_x, 27)
        .astype(np.float32)
    )
    return feats, avg_grid


def rect_features(master_rgb: Image.Image, box: Box) -> Tuple[np.ndarray, np.ndarray]:
    """27-dim LAB feature + average RGB for an arbitrary pixel rect."""
    region = master_rgb.crop(box)
    return subblock_lab(region), average_rgb(region)


def detail_energy(master_rgb: Image.Image, box: Box, sample: int = 32) -> float:
    """局部细节度量：亮度梯度均值 + 0.4×亮度标准差。

    与 DetailMap 同口径（噪点不误切、真实边缘可靠细分）。sample 是
    中间降采样分辨率，越小越平、越大越敏感。"""
    region = master_rgb.crop(box).convert("L").resize(
        (sample, sample), Image.Resampling.BOX
    )
    a = np.asarray(region, dtype=np.float32)
    gx = np.abs(np.diff(a, axis=1)).mean()
    gy = np.abs(np.diff(a, axis=0)).mean()
    return float(gx + gy + 0.4 * a.std())


class DetailMap:
    """细节能量积分图：O(1) 计算任意矩形的细节分。

    度量 = 亮度梯度均值 + 0.4 × 亮度标准差（三通道积分图：梯度幅值、
    亮度、亮度²）。相比纯梯度均值：噪点区（梯度虚高、std 低）不再误切，
    真实边缘/纹理（双高）被可靠细分。降采样到 ≤max_dim 保持大图可算。
    """

    def __init__(self, master: Image.Image, max_dim: int = 512):
        W, H = master.size
        f = max(1, int(max(W, H) / max_dim))
        small = master.convert("L").resize(
            (max(2, W // f), max(2, H // f)), Image.Resampling.BOX)
        a = np.asarray(small, dtype=np.float32)
        g = np.zeros_like(a)
        g[:, :-1] += np.abs(np.diff(a, axis=1))
        g[:-1, :] += np.abs(np.diff(a, axis=0))
        self.hh, self.hw = g.shape
        # 三通道：梯度幅值和、亮度和、亮度平方和
        self.I = np.zeros((self.hh + 1, self.hw + 1, 3), dtype=np.float64)
        chans = np.stack([g, a, a * a], axis=2)
        np.cumsum(np.cumsum(chans, axis=0), axis=1, out=self.I[1:, 1:])
        self.sx = W / self.hw
        self.sy = H / self.hh

    def _map(self, boxes: np.ndarray) -> np.ndarray:
        out = np.empty_like(boxes, dtype=np.int64)
        out[:, 0] = np.clip(np.floor(boxes[:, 0] / self.sx), 0, self.hw - 1)
        out[:, 1] = np.clip(np.floor(boxes[:, 1] / self.sy), 0, self.hh - 1)
        out[:, 2] = np.clip(np.ceil(boxes[:, 2] / self.sx), 1, self.hw)
        out[:, 3] = np.clip(np.ceil(boxes[:, 3] / self.sy), 1, self.hh)
        out[:, 2] = np.maximum(out[:, 2], out[:, 0] + 1)
        out[:, 3] = np.maximum(out[:, 3], out[:, 1] + 1)
        return out

    def energies(self, boxes) -> np.ndarray:
        """每矩形的细节分 = 梯度均值 + 0.4×亮度std，向量化。"""
        b = np.asarray(boxes, dtype=np.float64)
        if b.ndim == 1:
            b = b[None, :]
        c = self._map(b)
        x0, y0, x1, y1 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        sums = (self.I[y1, x1] - self.I[y0, x1] - self.I[y1, x0] + self.I[y0, x0])
        areas = np.maximum((x1 - x0) * (y1 - y0), 1)
        grad_mean = sums[:, 0] / areas
        lum_mean = sums[:, 1] / areas
        lum_sq_mean = sums[:, 2] / areas
        lum_std = np.sqrt(np.maximum(lum_sq_mean - lum_mean * lum_mean, 0.0))
        return grad_mean + 0.4 * lum_std

    def energy(self, box: Box) -> float:
        """单矩形细节分的标量快路径（零数组分配；叠加层逐格查询走它）。"""
        x0, y0, x1, y1 = box
        cx0 = int(np.clip(np.floor(x0 / self.sx), 0, self.hw - 1))
        cy0 = int(np.clip(np.floor(y0 / self.sy), 0, self.hh - 1))
        cx1 = int(np.clip(np.ceil(x1 / self.sx), 1, self.hw))
        cy1 = int(np.clip(np.ceil(y1 / self.sy), 1, self.hh))
        if cx1 <= cx0:
            cx1 = cx0 + 1
        if cy1 <= cy0:
            cy1 = cy0 + 1
        s = self.I[cy1, cx1] - self.I[cy0, cx1] - self.I[cy1, cx0] + self.I[cy0, cx0]
        area = max(1, (cx1 - cx0) * (cy1 - cy0))
        grad_mean = float(s[0] / area)
        lum_mean = float(s[1] / area)
        lum_sq_mean = float(s[2] / area)
        lum_std = (lum_sq_mean - lum_mean * lum_mean)
        return grad_mean + 0.4 * (lum_std if lum_std > 0 else 0.0) ** 0.5


class IntegralFeatures:
    """Integral images over a down-scaled copy: O(1) mean LAB/RGB per rect.

    Channels 0-2 = CIE LAB, 3-5 = RGB.  Sub-block means of any master rect
    (and its 3x3 subdivision) are exact averages over the downscaled pixels
    it covers — indistinguishable from full-res box averages in practice.
    """

    def __init__(self, master: Image.Image):
        W, H = master.size
        self.scale = 2 if W * H <= 4_000_000 else 4
        self.hw = max(3, W // self.scale)
        self.hh = max(3, H // self.scale)
        small = master.convert("RGB").resize((self.hw, self.hh), Image.Resampling.BOX)
        rgb = np.asarray(small, dtype=np.float64)
        lab = lab_array(small).astype(np.float64)
        chans = np.concatenate([lab, rgb], axis=2)             # (hh, hw, 6)
        self.I = np.zeros((self.hh + 1, self.hw + 1, 6), dtype=np.float64)
        np.cumsum(np.cumsum(chans, axis=0), axis=1, out=self.I[1:, 1:])
        self.sx = W / self.hw
        self.sy = H / self.hh

    def _map(self, boxes: np.ndarray) -> np.ndarray:
        """Master rects (L,4) -> downscaled integral coords (L,4) int.

        首边 floor、末边 ceil，保证矩形覆盖完整；末边 clip 到 hh/hw（积分图
        的有效末索引），首边再压到末边-1 —— 修复恰好落在图像底/右边缘的
        亚像素矩形把索引推到 hh+1 的越界。"""
        out = np.empty_like(boxes, dtype=np.int64)
        out[:, 0] = np.clip(np.floor(boxes[:, 0] / self.sx), 0, self.hw - 1)
        out[:, 1] = np.clip(np.floor(boxes[:, 1] / self.sy), 0, self.hh - 1)
        out[:, 2] = np.clip(np.ceil(boxes[:, 2] / self.sx), 1, self.hw)
        out[:, 3] = np.clip(np.ceil(boxes[:, 3] / self.sy), 1, self.hh)
        out[:, 2] = np.maximum(out[:, 2], out[:, 0] + 1)
        out[:, 3] = np.maximum(out[:, 3], out[:, 1] + 1)
        return out

    def _sums(self, boxes: np.ndarray) -> np.ndarray:
        """(L,4) master rects -> (L,6) channel sums."""
        c = self._map(boxes)
        x0, y0, x1, y1 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        return (self.I[y1, x1] - self.I[y0, x1] - self.I[y1, x0] + self.I[y0, x0])

    def rect_stats(self, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(L,4) master rects -> ((L,27) LAB features, (L,3) average RGB)."""
        boxes = np.asarray(boxes, dtype=np.float64)
        L = boxes.shape[0]
        x0, y0 = boxes[:, 0], boxes[:, 1]
        w = (boxes[:, 2] - boxes[:, 0]) / 3.0
        h = (boxes[:, 3] - boxes[:, 1]) / 3.0
        sub = np.empty((L * 9, 4), dtype=np.float64)
        i = 0
        for by in range(3):
            for bx in range(3):
                sub[i:i + L, 0] = x0 + bx * w
                sub[i:i + L, 1] = y0 + by * h
                sub[i:i + L, 2] = x0 + (bx + 1) * w
                sub[i:i + L, 3] = y0 + (by + 1) * h
                i += L
        sums = self._sums(sub)                                   # (9L, 6)
        c = self._map(sub)
        areas = ((c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1]))[:, None]
        means = sums / np.maximum(areas, 1)
        means = means.reshape(3, 3, L, 6).transpose(2, 0, 1, 3)  # (L,by,bx,6)
        feats = means[..., :3].reshape(L, 27).astype(np.float32)
        avgs = means[:, 0, 0, 3:6]
        full = self._sums(boxes) / np.maximum(
            ((self._map(boxes)[:, 2] - self._map(boxes)[:, 0])
             * (self._map(boxes)[:, 3] - self._map(boxes)[:, 1]))[:, None], 1)
        avgs = full[:, 3:6].astype(np.float32)
        return feats, avgs


def uniform_base_boxes(width: int, height: int, cells_x: int, cells_y: int) -> List[BaseBox]:
    cw = width / max(1, cells_x)
    ch = height / max(1, cells_y)
    return [
        ((int(gx * cw), int(gy * ch), int((gx + 1) * cw), int((gy + 1) * ch)), 1)
        for gy in range(cells_y)
        for gx in range(cells_x)
    ]


def _normalize_regions(regions) -> List[Tuple[int, int, int, int, int]]:
    out: List[Tuple[int, int, int, int, int]] = []
    for r in regions or []:
        if isinstance(r, dict):
            b = r["box"]
            level = int(r.get("level", 2))
            x0, y0, x1, y1 = (int(v) for v in b)
        else:
            x0, y0, x1, y1, level = (int(v) for v in r)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        out.append((x0, y0, x1, y1, max(1, level)))
    return out


def _intersects(box: Box, rx0: int, ry0: int, rx1: int, ry1: int) -> bool:
    return box[0] < rx1 and rx0 < box[2] and box[1] < ry1 and ry0 < box[3]


def adaptive_leaves(
    master_rgb: Image.Image,
    base_boxes: Sequence[BaseBox],
    levels: int,
    threshold: float,
    min_px: Optional[int] = None,
    regions=None,
    detail: Optional["DetailMap"] = None,
    first_level_factor: float = 1.0,
) -> List[Tuple[int, int, int, int, int]]:
    """Energy/region-driven refinement of arbitrary base boxes.

    min_px=None derives the smallest leaf from the base-cell size and depth:
    base_px / 2^(levels-1) — matching the classic proportional 1/2, 1/4, 1/8,
    1/16 nesting behaviour instead of an absolute pixel floor.

    first_level_factor > 1 makes first-level (base) cells harder to split
    (their effective threshold is threshold × first_level_factor) — the
    classic "keep large first-level blocks whole" bias.
    """
    regs = _normalize_regions(regions)
    leaves: List[Tuple[int, int, int, int, int]] = []
    energy_of = detail.energy if detail is not None else (
        lambda b: detail_energy(master_rgb, b))
    cap_needed = int(levels)
    if regs:
        cap_needed = max(cap_needed, max(r[4] for r in regs))
    if min_px is None and base_boxes:
        base_px = min(min((b[0][2] - b[0][0], b[0][3] - b[0][1]) for b in base_boxes))
        # smallest leaf edge = base_px / 2^(cap-1) (classic 1/2,1/4,1/8,1/16 nesting)
        min_px = max(1, base_px >> max(0, cap_needed - 1))
    elif min_px is None:
        min_px = 8

    def forced_level(box: Box) -> int:
        lv = 1
        for rx0, ry0, rx1, ry1, rl in regs:
            if _intersects(box, rx0, ry0, rx1, ry1):
                lv = max(lv, rl)
        return lv

    def rec(x0: int, y0: int, x1: int, y1: int, level: int) -> None:
        box = (x0, y0, x1, y1)
        forced = forced_level(box)
        cap = max(levels, forced)
        if level >= cap or (x1 - x0) <= min_px or (y1 - y0) <= min_px:
            leaves.append((x0, y0, x1, y1, level))
            return
        eff_threshold = threshold * (first_level_factor if level == 1 else 1.0)
        if level < forced or energy_of(box) > eff_threshold:
            mx = (x0 + x1) // 2
            my = (y0 + y1) // 2
            rec(x0, y0, mx, my, level + 1)
            rec(mx, y0, x1, my, level + 1)
            rec(x0, my, mx, y1, level + 1)
            rec(mx, my, x1, y1, level + 1)
            return
        leaves.append((x0, y0, x1, y1, level))

    for item in base_boxes:
        (x0, y0, x1, y1), lvl = item
        rec(x0, y0, x1, y1, lvl)
    return leaves


def export_poster(
    image: Image.Image, cols: int, rows: int, out_dir: str, stem: str,
    jpeg_quality: int = 100,
) -> List[str]:
    """Split a finished mosaic into poster pages (e.g. 3x2 or 2x3)."""
    os.makedirs(out_dir, exist_ok=True)
    W, H = image.size
    paths: List[str] = []
    for ry in range(rows):
        for cx in range(cols):
            box = (
                int(cx * W / cols), int(ry * H / rows),
                int((cx + 1) * W / cols), int((ry + 1) * H / rows),
            )
            path = os.path.join(out_dir, f"{stem}_r{ry + 1}c{cx + 1}.png")
            image.crop(box).save(path)
            paths.append(path)
    return paths


class _ThumbCache:
    """Bounded cache of decoded+resized(+blended) thumbnails."""

    def __init__(self, db: TileDB, capacity: int = 512):
        self.db = db
        self.capacity = capacity
        self._cache: Dict[Tuple[int, Tuple[int, int], int], Image.Image] = {}

    def get(self, tile_id: int, size: Tuple[int, int], blend_pct: int = 0,
            tint: Optional[Tuple[int, int, int]] = None) -> Image.Image:
        key = (tile_id, size, blend_pct if blend_pct and tint else 0)
        img = self._cache.get(key)
        if img is None:
            raw = self.db.get_thumb(tile_id)
            img = Image.open(io.BytesIO(raw))
            img = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            if blend_pct and tint is not None:
                solid = Image.new("RGB", size, tint)
                mixed = ImageChops.soft_light(img, solid)
                img = Image.blend(img, mixed, min(100, blend_pct) / 100.0)
            if len(self._cache) >= self.capacity:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = img
        return img


def _tile_groups(metas) -> np.ndarray:
    """Tile position -> source-photo id (all variants of a photo share one id)."""
    seen: Dict[object, int] = {}
    out = np.empty(len(metas), dtype=np.int32)
    for i, m in enumerate(metas):
        out[i] = seen.setdefault(m.get("md5", m.get("id", i)), len(seen))
    return out


def _decode_tile(raw: bytes, need: Tuple[int, int], src: Optional[tuple] = None) -> Image.Image:
    """Decode a tile once, only as large as its biggest leaf needs.

    Normally that is the stored thumbnail (JPEG draft mode decodes at 1/2, 1/4 or
    1/8 scale straight from the DCT).  When a leaf is *larger* than the stored
    thumbnail (high-resolution output) the original photo is re-read and cut
    exactly the way the thumbnail was (centre crop -> variant -> crop), so the
    big print is genuinely sharp instead of an up-scaled 256px thumbnail.  If the
    source file is gone the thumbnail is used as a fallback.
    """
    if src is not None:
        path, variant, tw, th = src
        if (need[0] > tw or need[1] > th) and os.path.exists(path):
            try:
                ratio = tw / float(max(1, th))
                with Image.open(path) as im:
                    im.draft("RGB", (max(need[0], need[1]),) * 2)
                    rgb = im.convert("RGB")
                base = VARIANTS[variant](crop_to_ratio(rgb, ratio))
                return crop_to_ratio(base, ratio)
            except Exception:
                pass
    im = Image.open(io.BytesIO(raw))
    im.draft("RGB", need)
    return im.convert("RGB")


def _fit_tile(img: Image.Image, size: Tuple[int, int], blend_pct: int,
              tint: Optional[Tuple[int, int, int]]) -> Image.Image:
    """Fit a decoded tile to one leaf, with optional soft-light colour tint."""
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    if blend_pct and tint is not None:
        solid = Image.new("RGB", size, tint)
        mixed = ImageChops.soft_light(img, solid)
        img = Image.blend(img, mixed, min(100, blend_pct) / 100.0)
    return img


def _render_tile(raw: bytes, size: Tuple[int, int], blend_pct: int,
                 tint: Optional[Tuple[int, int, int]], src: Optional[tuple] = None) -> Image.Image:
    return _fit_tile(_decode_tile(raw, size, src), size, blend_pct, tint)


def _paste_tiles(canvas: Image.Image, db: TileDB, jobs, blend_pct: int,
                 before=None, after=None, step: int = 2048, style=None) -> None:
    """Paste (leaf_index, x, y, size, tile_id, tint) jobs onto the canvas.

    Every distinct tile is fetched (batched SQL) and decoded exactly once - at the
    largest size any of its leaves needs - on a thread pool; each leaf then only
    pays a cheap resize/blend.  Pasting stays in leaf order so cancellation and
    checkpoints remain exact.
    """
    if not jobs:
        return
    need: Dict[int, Tuple[int, int]] = {}
    for j in jobs:
        w, h = j[3]
        o = need.get(j[4])
        need[j[4]] = (w, h) if o is None else (max(o[0], w), max(o[1], h))
    # a decoded tile is at most as big as its largest leaf (or the 256px thumbnail)
    est = sum(w * h * 3 for w, h in need.values())
    keep_all = est <= 600 * 1024 * 1024
    avg_bytes = max(1, est // max(1, len(need)))
    step = int(max(32, min(step, (400 * 1024 * 1024) // avg_bytes)))
    decoded: Dict[int, Image.Image] = {}
    workers = max(2, min(32, os.cpu_count() or 4))
    with ThreadPoolExecutor(workers) as ex:
        for s in range(0, len(jobs), step):
            chunk = jobs[s:s + step]
            todo = list({j[4] for j in chunk if j[4] not in decoded})
            if todo:
                raws = db.get_thumbs(todo)
                srcs = db.get_sources(todo)
                for tid, img in zip(todo, ex.map(
                        lambda tid: _decode_tile(raws[tid], need[tid], srcs.get(tid)), todo)):
                    decoded[tid] = img
                del raws
            for j in chunk:
                if before:
                    before(j[0])
                img = _fit_tile(decoded[j[4]], j[3], blend_pct, j[5])
                if style is not None and style.has_gaps():
                    m = style.mask(j[3])
                    canvas.paste(img, (j[1], j[2]), m)
                else:
                    canvas.paste(img, (j[1], j[2]))
                if after:
                    after(j[0])
            if not keep_all:
                decoded.clear()


def ext_is_raster(path: str) -> bool:
    """输出扩展名是否为非 TIFF 的光栅格式。"""
    return os.path.splitext(path)[1].lower() not in (".tif", ".tiff")


def _stream_output_path(out_path: str) -> str:
    """流式输出的扩展名路由：PNG/TIFF 保留，JPEG/BMP 改写为 PNG。

    PNG 与 BigTIFF 都有流式写入器；JPEG 格式上限 65535px 且 Pillow 只能
    整图编码，BMP 超 4GB 有格式限制——这两类统一改 PNG（无损且支持流式）。
    """
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".png", ".tif", ".tiff"):
        return out_path
    return os.path.splitext(out_path)[0] + ".png"


def _check_disk_space(out_path: str, est_bytes: Optional[int]) -> None:
    """流式渲染前的磁盘空间预检：预计输出放不下时尽早抛 ValueError。

    渲染可能耗时数小时，等写盘阶段才失败代价太高；不足时直接抛错，消息
    给出预计/剩余 GB 并建议缩小输出宽度或换盘（GUI 经 _on_failed 弹窗显示，
    CLI 直接打印）。仅流式路径调用——非流式整图输出由 Pillow 一次写出，
    维持原行为；ckpt 目录很小，占用忽略。磁盘用量查询失败（如特殊文件
    系统）时不拦截渲染。
    """
    if not est_bytes or est_bytes <= 0:
        return
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    try:
        free = shutil.disk_usage(out_dir).free
    except OSError:
        return
    if free < est_bytes:
        raise ValueError(
            f"磁盘空间不足：预计输出约 {est_bytes / (1 << 30):.2f} GB，"
            f"该目录当前仅剩 {free / (1 << 30):.2f} GB；"
            "请缩小输出宽度或换到剩余空间更大的磁盘再渲染。")


def make_mosaic(
    master_path: str,
    db: TileDB,
    out_path: str,
    cells_x: Optional[int] = None,
    cell_px: int = 64,
    tile_px: Optional[int] = None,
    top_k: int = 8,
    radius: Optional[float] = None,
    max_reuse: int = 0,
    seed: Optional[int] = None,
    jpeg_quality: int = 100,
    levels: int = 1,
    detail_threshold: float = 25.0,
    regions=None,
    leaf_boxes=None,
    blend: float = 0.0,
    device: str = "auto",
    cache_cap: int = 512,
    out_width: Optional[int] = None,
    checkpoint: Optional[str] = None,
    first_level_factor: float = 1.0,
    stream_threshold_px: int = STREAM_THRESHOLD_PX,
    stream_strip_rows: Optional[int] = None,
    style: str = "grid",
    signal: str = "detail",
    shared_matcher: Optional[Matcher] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    status: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Render the mosaic and save it. Returns run statistics.

    Resumability: for adaptive/manual modes a checkpoint directory (default
    ``<out>.ckpt``) stores a parameter fingerprint, the per-leaf tile picks and
    a canvas snapshot.  A re-run with identical parameters resumes matching and
    pasting where it stopped; a cancelled run keeps its progress.  The
    checkpoint is removed after a successful render.
    """
    with Image.open(master_path) as im:
        im.load()
        master = im.convert("RGB")
    W, H = master.size

    from .styles import get_style
    sty = get_style(style)

    if cells_x is None:
        cells_x = max(1, W // max(1, cell_px))
    cells_x = max(1, min(int(cells_x), W))
    cell_w = W / cells_x
    cells_y = max(1, int(round(H / cell_w)))

    tiles = db.load_arrays()
    # 批量场景可共享已建 Matcher（库只加载一次，GPU 特征常驻），由调用方持有
    if shared_matcher is None:
        matcher = Matcher(tiles, top_k=top_k, radius=radius, max_reuse=max_reuse,
                          seed=seed, device=device)
    else:
        matcher = shared_matcher
    ids = tiles["ids"]
    metas = tiles["metas"]
    blend_pct = int(round(max(0.0, min(1.0, float(blend))) * 100))
    cache = _ThumbCache(db, capacity=cache_cap)
    variant_counts: Dict[str, int] = {}

    manual = leaf_boxes is not None
    adaptive = (int(levels) > 1 or bool(regions))

    if manual or adaptive:
        if manual:
            base: List[BaseBox] = []
            for item in leaf_boxes:
                if isinstance(item, (tuple, list)) and len(item) == 2 \
                        and isinstance(item[0], (tuple, list)):
                    b, lvl = item
                else:
                    b, lvl = item, 1
                base.append(((int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                             int(lvl) if lvl else 1))
        else:
            base = uniform_base_boxes(W, H, cells_x, cells_y)
        if adaptive:
            # 细节信号：detail=梯度+对比度（纹理）；saliency=显著性主体（主旨优先）
            if signal == "saliency":
                from .saliency import SaliencyMap
                integral_detail = SaliencyMap(master)
            else:
                integral_detail = DetailMap(master)
            leaves = adaptive_leaves(master, base, int(levels),
                                     float(detail_threshold), regions=regions,
                                     detail=integral_detail,
                                     first_level_factor=first_level_factor)
        else:
            leaves = [(b[0], b[1], b[2], b[3], lvl) for (b, lvl) in base]
        total = len(leaves)
        boxes = np.asarray([l[:4] for l in leaves], dtype=np.float64)
        integral = IntegralFeatures(master)
        feats, avgs = integral.rect_stats(boxes)
        del integral

        # ---- checkpoint bookkeeping ------------------------------------
        ckpt_dir = checkpoint if checkpoint else (out_path + ".ckpt")
        meta_path = os.path.join(ckpt_dir, "meta.json")
        canvas_path = os.path.join(ckpt_dir, "canvas.png")
        try:
            mstat = os.stat(master_path)
            master_sig = [master_path, mstat.st_size, int(mstat.st_mtime)]
        except OSError:
            master_sig = [master_path]
        key = hashlib.sha256(json.dumps(
            {"master": master_sig, "db": [db.path, db.count()],
             "params": [cells_x, levels, detail_threshold, regions, top_k,
                        radius, max_reuse, seed, blend, W, H, out_width],
             "leaves": [list(l) for l in leaves]},
            sort_keys=True, default=str).encode("utf-8")).hexdigest()

        # optional output resolution (print-size export): canvas scaled from master
        s = 1.0
        if out_width:
            s = max(1e-3, float(out_width) / W)
        OW, OH = int(round(W * s)), int(round(H * s))

        # 超大输出：分条带流式渲染（BigTIFF），画布不再整图常驻内存
        from .streaming import choose_strip_rows, estimate_streaming_size  # noqa

        stream = OW * OH > stream_threshold_px
        if stream:
            # 输出格式路由：PNG 流式直写（不可续传）；TIFF→BigTIFF（可续传）；
            # JPEG/BMP 不支持流式（且 JPEG 格式上限 65535px）→ 改 PNG
            out_path = _stream_output_path(out_path)
            rps = choose_strip_rows(OW, OH)
            est_bytes = estimate_streaming_size(OW, OH, rps)
            _check_disk_space(out_path, est_bytes)   # 写盘前预检：不足直接失败
            ckpt_on = False              # 条带级续传由流式路径自己负责
        else:
            est_bytes = None
            ckpt_on = OW * OH <= 80_000_000  # snapshots of a giant canvas cost more than they save

        picks: List[Optional[int]] = [None] * total
        matched = 0
        pasted = 0
        resumed = False
        if stream:
            pass                                # 条带续传在流式渲染内部处理
        else:
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    if (meta.get("key") == key and meta.get("total") == total
                            and os.path.exists(canvas_path)):
                        picks = list(meta["picks"])
                        matched = int(meta["matched"])
                        pasted = int(meta["pasted"])
                        resumed = True
                except Exception:
                    resumed = False
            if resumed:
                canvas = Image.open(canvas_path).convert("RGB")
            else:
                # 有缝隙的风格（蜂窝/点彩）垫高斯模糊主图，否则白底
                canvas = (sty.background(master, (OW, OH)) if sty.has_gaps()
                          else Image.new("RGB", (OW, OH), (255, 255, 255)))
                picks = [None] * total
                matched = pasted = 0

        def save_ckpt(matched_n: int, pasted_n: int) -> None:
            if stream or not ckpt_on:
                return
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"key": key, "total": total, "picks": picks,
                           "matched": matched_n, "pasted": pasted_n}, fh)
            canvas.save(canvas_path)

        # ---- matching (only for leaves without a pick yet) --------------
        def say(msg: str) -> None:
            if status:
                status(msg)

        def check() -> None:
            if should_cancel and should_cancel():
                save_ckpt(matched, pasted)
                raise RenderCancelled("cancelled during matching")

        undone = [i for i in range(total) if picks[i] is None]
        dists = np.zeros(total, dtype=np.float64)
        mean_dist = 0.0
        if undone:
            check()
            groups = _tile_groups(metas)
            phase_seen: List[str] = []

            def on_plan(phase: str, d: int, tot: int) -> None:
                if not phase_seen or phase_seen[-1] != phase:
                    phase_seen.append(phase)
                    say("正在筛选候选瓦片…" if phase == "match" else "正在智能分配瓦片…")
                if progress:
                    progress(d, tot)

            up, ud = plan_assignments(
                matcher, np.asarray(boxes, dtype=np.int64)[undone], feats[undone],
                avgs[undone], groups, (W, H), top_k=top_k, seed=seed,
                max_reuse=max_reuse, progress=on_plan, check=check)
            for j, i in enumerate(undone):
                picks[i] = int(up[j])
                dists[i] = float(ud[j])
            matched = total
            mean_dist = float(np.mean(ud))
            if shared_matcher is None:      # 共享 Matcher 由调用方持有，不释放
                matcher.release()
        del feats

        # ---- pasting (parallel decode / fit / blend, ordered paste) --------
        say("正在拼贴…")
        histogram: Dict[int, int] = {}
        leaf_rects: List[Tuple[int, int, int, int, int]] = []
        chosen: List[int] = []
        jobs = []
        stream_info: Dict[str, object] = {}
        for i, leaf in enumerate(leaves):
            x0, y0, x1, y1, level = leaf
            histogram[level] = histogram.get(level, 0) + 1
            # leaf rect in OUTPUT coordinates (canvas may be up-scaled)
            ox0, oy0 = int(round(x0 * s)), int(round(y0 * s))
            ox1, oy1 = int(round(x1 * s)), int(round(y1 * s))
            leaf_rects.append((ox0, oy0, ox1, oy1, level))
            tid = int(ids[picks[i]])
            chosen.append(tid)
            if not stream and i < pasted:
                continue                      # already on the resumed canvas
            variant = metas[picks[i]]["variant"]
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
            if stream:
                continue                      # 流式路径按条带自行生成贴图任务
            jobs.append((i, ox0, oy0, (max(1, ox1 - ox0), max(1, oy1 - oy0)), tid,
                         (int(avgs[i][0]), int(avgs[i][1]), int(avgs[i][2]))))
        last_ckpt = [time.monotonic()]

        if stream:
            # ---- 流式：条带渲染直写 BigTIFF（内存占用与输出总面积解耦）----
            from .streaming import render_streaming

            preview_path = os.path.splitext(out_path)[0] + ".preview.png"
            meta_strips_path = os.path.join(ckpt_dir, "meta.json")

            def save_strips(n: int) -> None:
                os.makedirs(ckpt_dir, exist_ok=True)
                with open(meta_strips_path, "w", encoding="utf-8") as fh:
                    json.dump({"key": key, "done_strips": n}, fh)

            done_strips = 0
            if os.path.exists(meta_strips_path):
                try:
                    with open(meta_strips_path, "r", encoding="utf-8") as fh:
                        m2 = json.load(fh)
                    if m2.get("key") == key:
                        done_strips = int(m2.get("done_strips", 0))
                except Exception:
                    done_strips = 0

            stream_info = render_streaming(
                leaves, picks, ids, metas, db, out_path, OW, OH, s,
                blend_pct=blend_pct, avgs=avgs,
                preview_path=preview_path,
                should_cancel=should_cancel, progress=progress, status=status,
                done_strips=done_strips, ckpt_strips=save_strips,
                strip_rows=stream_strip_rows, style=sty, master=master)
            del avgs
        else:
            def before(i: int) -> None:
                if should_cancel and should_cancel():
                    save_ckpt(matched, i)
                    raise RenderCancelled("cancelled during paste")

            def after(i: int) -> None:
                if progress and ((i + 1) % 256 == 0 or i + 1 == total):
                    progress(i + 1, total)
                if time.monotonic() - last_ckpt[0] > 60:      # snapshot at most once a minute
                    save_ckpt(matched, i + 1)
                    last_ckpt[0] = time.monotonic()

            _paste_tiles(canvas, db, jobs, blend_pct, before, after, style=sty)
            del avgs

            ext_ = os.path.splitext(out_path)[1].lower()
            if ext_ in (".jpg", ".jpeg"):
                canvas.save(out_path, "JPEG", quality=jpeg_quality)
            elif ext_ == ".bmp":
                canvas.save(out_path, "BMP")
            elif ext_ in (".tif", ".tiff"):
                canvas.save(out_path, "TIFF")
            else:
                canvas.save(out_path)
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        mode = "manual" if manual and not adaptive else "adaptive"
        stats_extra: Dict[str, object] = {
            "mode": mode,
            "leaves": total,
            "level_histogram": histogram,
            "leaf_rects": leaf_rects,
            "choices": chosen,
            "mean_distance": mean_dist,
            "resumed_pasted": pasted if resumed else 0,
            "device": matcher.device,
        }
        if stream:
            stats_extra.update({
                "streamed": True,
                "output_bytes": stream_info.get("bytes"),
                "preview": stream_info.get("preview", ""),
                "strips": stream_info.get("strips"),
                "leaf_rects": stream_info.get("leaf_rects", leaf_rects),
                "choices": stream_info.get("choices", chosen),
                "level_histogram": stream_info.get("level_histogram", histogram),
            })
            if est_bytes:
                stats_extra["est_output_bytes"] = est_bytes
        out_size = (OW, OH)
        return {
            "cells_x": cells_x,
            "cells_y": cells_y,
            "tile_count": matcher.n,
            "unique_used": int(np.count_nonzero(matcher.usage)),
            "variant_counts": variant_counts,
            "output": os.path.abspath(out_path),
            "size": out_size,
            **stats_extra,
        }
    else:
        feats, avgs = master_cell_features(master, cells_x, cells_y)
        flat_f = feats.reshape(-1, 27)
        flat_a = avgs.reshape(-1, 3)
        out_px = int(tile_px or (max(1, round(out_width / cells_x)) if out_width else cell_px))
        total = cells_x * cells_y
        gy, gx = np.divmod(np.arange(total, dtype=np.int64), cells_x)
        boxes_u = np.stack([gx * out_px, gy * out_px, (gx + 1) * out_px,
                            (gy + 1) * out_px], axis=1)
        if status:
            status("正在智能分配瓦片…")
        picks, dists = plan_assignments(
            matcher, boxes_u, flat_f, flat_a, _tile_groups(metas),
            (cells_x * out_px, cells_y * out_px), top_k=top_k, seed=seed,
            max_reuse=max_reuse,
            progress=(lambda ph, d, tot: progress(d, tot)) if progress else None,
            check=(lambda: (_ for _ in ()).throw(RenderCancelled("cancelled"))
                   if (should_cancel and should_cancel()) else None))
        if shared_matcher is None:          # 共享 Matcher 由调用方持有，不释放
            matcher.release()
        OW_u, OH_u = cells_x * out_px, cells_y * out_px

        if OW_u * OH_u > stream_threshold_px:
            # ---- 超大 uniform 输出：同样走分条带流式渲染（画布不整图常驻）----
            # uniform 没有 leaves 概念：把输出坐标网格 boxes_u 直接构造成叶子
            # 列表（s=1.0，叶子坐标即输出坐标），picks/flat_a 与整图路径完全
            # 同源——条带按相交区裁剪后与整图路径逐像素一致。
            from .streaming import (choose_strip_rows, estimate_streaming_size,
                                    render_streaming)

            out_path = _stream_output_path(out_path)
            est_bytes_u = estimate_streaming_size(
                OW_u, OH_u, choose_strip_rows(OW_u, OH_u))
            _check_disk_space(out_path, est_bytes_u)

            # 条带级续传：与 adaptive 分支同模式（meta.json 存 key + done_strips）
            ckpt_dir = checkpoint if checkpoint else (out_path + ".ckpt")
            meta_strips_path = os.path.join(ckpt_dir, "meta.json")
            try:
                mstat = os.stat(master_path)
                master_sig = [master_path, mstat.st_size, int(mstat.st_mtime)]
            except OSError:
                master_sig = [master_path]
            key = hashlib.sha256(json.dumps(
                {"master": master_sig, "db": [db.path, db.count()],
                 "params": [cells_x, cells_y, out_px, top_k, radius, max_reuse,
                            seed, blend, OW_u, OH_u, "uniform-stream"]},
                sort_keys=True, default=str).encode("utf-8")).hexdigest()

            def save_strips(n: int) -> None:
                os.makedirs(ckpt_dir, exist_ok=True)
                with open(meta_strips_path, "w", encoding="utf-8") as fh:
                    json.dump({"key": key, "done_strips": n}, fh)

            done_strips = 0
            if os.path.exists(meta_strips_path):
                try:
                    with open(meta_strips_path, "r", encoding="utf-8") as fh:
                        m2 = json.load(fh)
                    if m2.get("key") == key:
                        done_strips = int(m2.get("done_strips", 0))
                except Exception:
                    done_strips = 0

            leaves_u: List[Tuple[int, int, int, int, int]] = []
            for i in range(total):
                bx = boxes_u[i]
                leaves_u.append((int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]), 1))
                pos = int(picks[i])
                variant = metas[pos]["variant"]
                variant_counts[variant] = variant_counts.get(variant, 0) + 1

            preview_path = os.path.splitext(out_path)[0] + ".preview.png"
            if status:
                status("正在拼贴…")
            stream_info = render_streaming(
                leaves_u, picks, ids, metas, db, out_path, OW_u, OH_u, 1.0,
                blend_pct=blend_pct, avgs=flat_a,
                preview_path=preview_path,
                should_cancel=should_cancel, progress=progress, status=status,
                done_strips=done_strips, ckpt_strips=save_strips,
                strip_rows=stream_strip_rows, style=sty, master=master)
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            stats_extra: Dict[str, object] = {
                "mode": "uniform",
                "leaves": total,
                "level_histogram": stream_info["level_histogram"],
                "leaf_rects": stream_info["leaf_rects"],
                "choices": stream_info["choices"],
                "mean_distance": float(np.mean(dists)) if total else 0.0,
                "resumed_pasted": 0,
                "streamed": True,
                "output_bytes": stream_info.get("bytes"),
                "preview": stream_info.get("preview", ""),
                "strips": stream_info.get("strips"),
                "est_output_bytes": est_bytes_u,
            }
            return {
                "cells_x": cells_x,
                "cells_y": cells_y,
                "tile_count": matcher.n,
                "device": matcher.device,
                "unique_used": int(np.count_nonzero(matcher.usage)),
                "variant_counts": variant_counts,
                "output": os.path.abspath(out_path),
                "size": (OW_u, OH_u),
                **stats_extra,
            }
        # 有缝隙的风格垫高斯模糊主图；砖墙风格偶数行右移半格
        canvas = (sty.background(master, (cells_x * out_px, cells_y * out_px))
                  if sty.has_gaps()
                  else Image.new("RGB", (cells_x * out_px, cells_y * out_px),
                                 (255, 255, 255)))
        leaf_rects = []
        chosen = []
        jobs = []
        for idx in range(total):
            y, x = divmod(idx, cells_x)
            pos = int(picks[idx])
            variant = metas[pos]["variant"]
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
            tid = int(ids[pos])
            leaf_rects.append((x * out_px, y * out_px, (x + 1) * out_px,
                               (y + 1) * out_px, 1))
            chosen.append(tid)
            dx, dy = sty.offset(y, x, out_px, out_px)      # 砖墙：偶数行右移半格
            jobs.append((idx, x * out_px + dx, y * out_px + dy, (out_px, out_px), tid,
                         (int(flat_a[idx][0]), int(flat_a[idx][1]), int(flat_a[idx][2]))))
        if status:
            status("正在拼贴…")
        _paste_tiles(canvas, db, jobs, blend_pct, style=sty,
                     before=(lambda i: (_ for _ in ()).throw(RenderCancelled("cancelled"))
                             if (should_cancel and should_cancel()) else None),
                     after=(lambda i: progress(i + 1, total)
                            if progress and ((i + 1) % 256 == 0 or i + 1 == total) else None))
        stats_extra = {"mode": "uniform", "leaves": total,
                       "level_histogram": {1: total},
                       "leaf_rects": leaf_rects, "choices": chosen,
                       "mean_distance": float(np.mean(dists)) if total else 0.0,
                       "resumed_pasted": 0}
        out_size = canvas.size

    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        canvas.save(out_path, "JPEG", quality=jpeg_quality)
    elif ext == ".bmp":
        canvas.save(out_path, "BMP")
    elif ext in (".tif", ".tiff"):
        canvas.save(out_path, "TIFF")
    else:
        canvas.save(out_path)

    return {
        "cells_x": cells_x,
        "cells_y": cells_y,
        "tile_count": matcher.n,
        "device": matcher.device,
        "unique_used": int(np.count_nonzero(matcher.usage)),
        "variant_counts": variant_counts,
        "output": os.path.abspath(out_path),
        "size": out_size,
        **stats_extra,
    }
