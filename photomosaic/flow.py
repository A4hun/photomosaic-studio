"""流体马赛克：照片沿主图轮廓方向流动排列——彻底没有网格。

思路（"不被困住的思路"之 1+2）：
1. 结构张量方向场：主图每个位置的"边缘走向"（梯度垂直方向），
   高斯平滑后的结构张量主方向——纹理区也有稳定方向，纯 numpy/PIL。
2. 流线追踪：从抖动种子行出发沿方向场积分，每隔约一个瓦片长放一张瓦片，
   瓦片旋转=局部流向、沿流向拉伸——照片像水流一样贴着主图的边走。
3. 选片：瓦片位置底下主图像素块的 27 维 LAB 特征 → 复用现有 Matcher。
4. 边界溶解：旋转贴图带羽化 alpha（圆角矩形蒙版+高斯模糊），
   相邻照片互相溶解过渡，近看是连续的照片河。
背景 = 主图高斯模糊（缝隙也读得出图）。
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .features import average_rgb, subblock_lab


def _blur_f(arr: np.ndarray, radius: float) -> np.ndarray:
    """对正值浮点阵做高斯模糊（uint8 缩放近似，方向场足够，无需 scipy）。"""
    a = np.clip(arr, 0.0, None)
    scale = max(float(a.max()), 1e-9) / 255.0
    img = Image.fromarray(np.clip(a / scale, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)),
                      dtype=np.float64) * scale


def orientation_field(master: Image.Image, max_dim: int = 512
                      ) -> Tuple[np.ndarray, float]:
    """结构张量方向场：每像素"边缘走向"角（弧度，(H,W)），及场/输出缩放比。"""
    im = master.convert("L")
    s = min(1.0, max_dim / max(im.size))
    W2 = max(8, int(im.width * s))
    H2 = max(8, int(im.height * s))
    small = im.resize((W2, H2), Image.Resampling.BOX)
    a = np.asarray(small, dtype=np.float64)
    gy, gx = np.gradient(a)
    Jxx = _blur_f(gx * gx, 3.0)
    Jxy = _blur_f(gx * gy, 3.0)
    Jyy = _blur_f(gy * gy, 3.0)
    # 结构张量主方向 = 梯度方向（垂直于边）；沿边方向 = 主方向 + 90°
    theta_major = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    theta = theta_major + np.pi / 2
    # 平坦区（无结构）：方向未定义 → 回落 0°（水平静流，别竖着拉丝）
    theta = np.where((Jxx + Jyy) > 1e-6, theta, 0.0)
    return theta, s


def flow_placements(theta: np.ndarray, field_scale: float,
                    out_w: int, out_h: int,
                    tile_len: float, tile_wid: float,
                    seed: Optional[int], step: float = 0.95
                    ) -> List[Tuple[float, float, float]]:
    """沿方向场追踪流线，每隔 step×瓦片长放一张（行间相位交错像并排水流）。"""
    H, W = theta.shape
    rng = np.random.default_rng(seed)
    placements: List[Tuple[float, float, float]] = []
    rows = max(2, int(out_h / max(4.0, tile_wid)))
    min_gap = tile_len * 0.9                     # 同一条流线上两瓦片的最小间距
    for r in range(rows):
        y = (r + 0.5) * out_h / rows + rng.uniform(-0.3, 0.3) * out_h / rows
        x = -1.5 * tile_len                       # 起点固定在左边界外一点
        # 行间相位交错：把"上一张"虚拟放在随机更左处 → 首张落点随机错开
        last: Optional[Tuple[float, float]] = (
            x - rng.uniform(0.0, 1.0) * tile_len, y)
        guard = 0
        while -1.6 * tile_len <= x <= out_w + tile_len and \
                -tile_len <= y <= out_h + tile_len:
            guard += 1
            if guard > 20000:
                break
            fx = min(W - 1, max(0, int(x * field_scale)))
            fy = min(H - 1, max(0, int(y * field_scale)))
            t = float(theta[fy, fx])
            if math.cos(t) < 0:                       # 统一前进向 = 向右
                t += math.pi
            if last is None or math.hypot(x - last[0], y - last[1]) >= min_gap:
                placements.append((x, y, t))
                last = (x, y)
            x += math.cos(t) * tile_len * step
            y += math.sin(t) * tile_len * step
            # 上下环绕（流场标准做法）：垂直流向的流线不会走出界就死
            if y < -tile_len:
                y += out_h + 2 * tile_len
                last = None                       # 环绕后距离基准重置
            elif y > out_h + tile_len:
                y -= out_h + 2 * tile_len
                last = None
    return placements


def render_flow(master: Image.Image, db, out_path: str, matcher=None,
                tile_len_frac: float = 0.05, aspect: float = 1.7,
                blend_pct: int = 45, seed: Optional[int] = None,
                jpeg_quality: int = 100, device: str = "auto",
                feather: float = 0.18,
                status=None, should_cancel=None, progress=None
                ) -> Dict[str, object]:
    """渲染流体马赛克（整图内存路径；stats 形状与 make_mosaic 兼容）。"""
    from .render import Matcher, RenderCancelled, _decode_tile, _fit_tile

    W, H = master.size
    tile_len = max(12.0, tile_len_frac * min(W, H))     # 沿流向长
    tile_wid = max(8.0, tile_len / max(1.05, aspect))   # 垂直流向宽

    if status:
        status("流体：计算轮廓方向场…")
    theta, fs = orientation_field(master)
    if status:
        status("流体：沿轮廓追踪流线…")
    placements = flow_placements(theta, fs, W, H, tile_len, tile_wid, seed)
    if not placements:
        raise ValueError("流体布局为空：主图太小或参数异常")

    tiles = db.load_arrays()
    ids = tiles["ids"]
    if matcher is None:
        matcher = Matcher(tiles, top_k=1, device=device)
        own_matcher = True
    else:
        own_matcher = False

    # 每个位置的取样块（中心方形，边=瓦片宽）→ 27 维特征 + 均色
    if status:
        status(f"流体：采样 {len(placements)} 个位置…")
    feats = np.empty((len(placements), 27), dtype=np.float32)
    avgs = np.empty((len(placements), 3), dtype=np.float32)
    patch = max(4, int(round(tile_wid)))
    for i, (cx, cy, _a) in enumerate(placements):
        x0 = min(max(0, int(cx - patch / 2)), W - patch)
        y0 = min(max(0, int(cy - patch / 2)), H - patch)
        region = master.crop((x0, y0, x0 + patch, y0 + patch))
        feats[i] = subblock_lab(region)
        avgs[i] = average_rgb(region)

    if status:
        status("流体：选片…")
    if progress:
        progress(0, len(placements))
    pos, d = matcher.select_batch(feats, avgs)
    tids = [int(ids[int(p)]) for p in np.asarray(pos)]

    if status:
        status("流体：贴片（溶解边界）…")
    # 背景 = 主图高斯模糊（缝隙读得出图，照片河浮在图上）
    canvas = master.convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=max(2, min(W, H) // 100)))
    raws = db.get_thumbs(tids)
    srcs = db.get_sources(tids)
    need = (int(round(tile_len)), int(round(tile_wid)))
    # 羽化蒙版：圆角矩形 + 高斯 → 溶解边界
    m0 = Image.new("L", need, 0)
    ImageDraw.Draw(m0).rounded_rectangle(
        (0, 0, need[0] - 1, need[1] - 1), radius=int(need[1] * 0.35), fill=255)
    m0 = m0.filter(ImageFilter.GaussianBlur(radius=max(1.0, feather * need[1])))

    cov_img = Image.new("L", (W, H), 0)             # 真实覆盖图（蒙版累积）
    for i, (cx, cy, ang) in enumerate(placements):
        if should_cancel and should_cancel():
            if own_matcher:
                matcher.release()
            raise RenderCancelled("cancelled during flow paste")
        if progress and i % 64 == 0:
            progress(i, len(placements))
        tile = _decode_tile(raws[tids[i]], need, srcs.get(tids[i]))
        tile = _fit_tile(tile, need, blend_pct,
                         (int(avgs[i][0]), int(avgs[i][1]), int(avgs[i][2])))
        deg = math.degrees(ang)
        tile = tile.rotate(deg, resample=Image.Resampling.BICUBIC, expand=True)
        mm = m0.rotate(deg, resample=Image.Resampling.BICUBIC, expand=True)
        pos = (int(cx - tile.width / 2), int(cy - tile.height / 2))
        canvas.paste(tile, pos, mm)
        cov_img.paste(255, pos, mm)

    # ---- 补空（保守）：只补真正的大空洞，且稀疏 ----------------------------
    # 教训：密集补空=随机位置乱堆，非常丑。改成：该格及周边 2 格半径都几乎
    # 没被碰到才算空洞；每格留一半跳过（稀疏点缀）；总量封顶流线数的 1/3。
    Hf, Wf = theta.shape
    cell = max(2.0, tile_wid)                       # 补空扫描步距
    gw, gh = int(W / cell) + 1, int(H / cell) + 1
    cov_small = np.asarray(
        cov_img.resize((gw, gh), Image.Resampling.BOX), dtype=np.float64)
    uncovered = cov_small < 6.0                     # 几乎绝对没被瓦片碰到
    # 腐蚀一圈：周边 2 格也没碰到才算真空洞
    padded = np.pad(uncovered, 2, constant_values=False)
    eroded = uncovered.copy()
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            if dx or dy:
                eroded &= padded[2 + dy:2 + dy + gh, 2 + dx:2 + dx + gw]
    cap = max(24, len(placements) // 3)
    fills: List[Tuple[float, float, float]] = []
    fy_rng = np.random.default_rng(None if seed is None else seed + 1)
    for gy in range(gh):
        for gx in range(gw):
            if len(fills) >= cap:
                break
            if not eroded[gy, gx] or (gx + gy) % 2:
                continue                            # 稀疏：隔一格补一张
            cx = (gx + 0.5 + fy_rng.uniform(-0.3, 0.3)) * cell
            cy = (gy + 0.5 + fy_rng.uniform(-0.3, 0.3)) * cell
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            fx = min(Wf - 1, max(0, int(cx * fs)))
            fy2 = min(Hf - 1, max(0, int(cy * fs)))
            fills.append((cx, cy, float(theta[fy2, fx])))
    if fills:
        if status:
            status(f"流体：补空 {len(fills)} 张…")
        ffeats = np.empty((len(fills), 27), dtype=np.float32)
        favgs = np.empty((len(fills), 3), dtype=np.float32)
        for i, (cx, cy, _a) in enumerate(fills):
            x0 = min(max(0, int(cx - patch / 2)), W - patch)
            y0 = min(max(0, int(cy - patch / 2)), H - patch)
            region = master.crop((x0, y0, x0 + patch, y0 + patch))
            ffeats[i] = subblock_lab(region)
            favgs[i] = average_rgb(region)
        fpos, _fd = matcher.select_batch(ffeats, favgs)
        for i, (cx, cy, ang) in enumerate(fills):
            tid = int(ids[int(fpos[i])])
            if tid not in raws:
                raws[tid] = db.get_thumbs([tid])[tid]
                srcs[tid] = db.get_sources([tid])[tid]
            tile = _decode_tile(raws[tid], need, srcs.get(tid))
            tile = _fit_tile(tile, need, blend_pct,
                             (int(favgs[i][0]), int(favgs[i][1]), int(favgs[i][2])))
            deg = math.degrees(ang)
            tile = tile.rotate(deg, resample=Image.Resampling.BICUBIC, expand=True)
            mm = m0.rotate(deg, resample=Image.Resampling.BICUBIC, expand=True)
            canvas.paste(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)), mm)
        placements.extend(fills)
        tids.extend(int(ids[int(p)]) for p in fpos)
    if own_matcher:
        matcher.release()
    if progress:
        progress(len(placements), len(placements))

    ext_ = os.path.splitext(out_path)[1].lower()
    if ext_ in (".jpg", ".jpeg"):
        canvas.save(out_path, "JPEG", quality=jpeg_quality)
    elif ext_ == ".bmp":
        canvas.save(out_path, "BMP")
    else:
        canvas.save(out_path)
    return {
        "mode": "flow",
        "leaves": len(placements),
        "tile_count": matcher.n,
        "unique_used": len(set(tids)),
        "variant_counts": {},
        "output": os.path.abspath(out_path),
        "size": canvas.size,
        "mean_distance": float(np.mean(d)) if len(d) else 0.0,
        "level_histogram": {1: len(placements)},
        "choices": tids,
        "leaf_rects": [],
    }
