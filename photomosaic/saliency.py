"""显著性检测（saliency detection）：找出图像的视觉重点/主体区域。

两路经典算法融合（纯 numpy/FFT，零新依赖、离线可用）：
1. **区域颜色对比度（HC，Cheng 2011）**：每个颜色与全图颜色分布的对比
   （按出现频率加权）。纯色主体在纹理背景上→高显著（颜色独特）；
   灰色噪块互相比→不显著（颜色同质）。与内部纹理无关——这是
   "抓主体"的主力。
2. **频谱残差（Hou & Zhang 2007）**：FFT 幅度谱的"惊奇"成分。纹理主体
   /高对比边缘→高显著。作为互补（HC 管颜色独特，谱残差管结构惊奇）。

融合 = 0.6×HC + 0.4×谱残差（各自 P95 稳健归一化后加权），
再盒式模糊聚成连贯块（不碎成噪点）。
诊断基准（合成已知掩码，Top-30% 面积的 IoU）：
纯色主体+纹理底 7%→≥60%（谱残差单路的致命盲区，HC 修复）；
偏离中心主体 27%→≥50%；纹理主体+纯色底 85%→保持。

返回 0-1 浮点显著性图（越大越"主旨"）。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """盒式模糊（积分图实现，边缘重复填充）。纯 numpy，无 PIL 浮点限制。"""
    r = max(1, int(radius))
    p = np.pad(a, r, mode="edge")
    I = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    np.cumsum(np.cumsum(p, axis=0), axis=1, out=I[1:, 1:])
    k = 2 * r + 1
    return (I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]) / float(k * k)


def _robust01(x: np.ndarray) -> np.ndarray:
    """P95 稳健归一化（主体永远凸显，与整体对比度无关；地板防平坦图放大）。"""
    p95 = float(np.percentile(x, 95))
    return x / max(p95, 1e-6)


def _histogram_contrast(rgb: np.ndarray, bits: int = 4) -> np.ndarray:
    """HC 颜色对比显著性（Cheng 2011 的直方图版）。

    颜色按 bits 位/通道量化；每个量化色的显著度 = 与全图所有颜色
    （按频率加权）的距离和。罕见且独特的颜色区域→显著。"""
    h, w, _ = rgb.shape
    q = (rgb >> (8 - bits)).astype(np.int64)
    nbin = 1 << (3 * bits)
    idx = (q[..., 0] << (2 * bits)) | (q[..., 1] << bits) | q[..., 2]
    hist = np.bincount(idx.ravel(), minlength=nbin).astype(np.float64)
    total = max(1.0, hist.sum())
    p = hist / total
    used = hist > 0
    # 每个实际出现色的代表色（bin 中心，直接由量化值重建）
    qv = np.arange(nbin)
    cols = np.empty((nbin, 3), dtype=np.float64)
    cols[:, 0] = (qv >> (2 * bits)) & ((1 << bits) - 1)
    cols[:, 1] = (qv >> bits) & ((1 << bits) - 1)
    cols[:, 2] = qv & ((1 << bits) - 1)
    cols *= (255.0 / ((1 << bits) - 1))
    # 只对出现过的色算（bin 数 ≤ 4096，全矩阵也才 16M 距离）
    sal_bin = np.zeros(nbin, dtype=np.float64)
    if used.sum() > 1:
        ui = np.flatnonzero(used)
        c = cols[ui]                                   # (u, 3)
        pu = p[ui]                                     # (u,)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))   # (u, u)
        sal = (d * pu[None, :]).sum(axis=1)            # Σ_b p(b)·|c_i−c_b|
        sal_bin[ui] = sal
    out = sal_bin[idx]
    return out.reshape(h, w)


def saliency_map(master: Image.Image, max_dim: int = 256) -> np.ndarray:
    """返回 (h, w) 的 0-1 显著性图（降采样到 max_dim 内，保持比例）。

    HC 颜色对比（0.6）+ 谱残差（0.4）融合，各自 P95 归一化。"""
    W, H = master.size
    f = max(1, int(max(W, H) / max_dim))
    w, h = max(2, W // f), max(2, H // f)
    small = master.convert("RGB").resize((w, h), Image.Resampling.BOX)
    rgb = np.asarray(small, dtype=np.uint8)
    gray = rgb.astype(np.float64).mean(axis=2)

    # ---- 1) HC 颜色对比（主力：抓颜色独特的主体，无视纹理） ----------
    hc = _histogram_contrast(rgb, bits=4)
    hc = _box_blur(hc, max(1, min(w, h) // 48))        # 聚成连贯区域

    # ---- 2) 频谱残差（互补：抓结构惊奇/纹理主体） ----------------------
    spec = np.fft.fft2(gray)
    amp = np.abs(spec)
    phase = np.angle(spec)
    log_amp = np.log(amp + 1e-9)
    mean = _box_blur(log_amp, 2)
    residual = log_amp - mean
    sal_spec = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal_spec = _box_blur(sal_spec, max(1, min(w, h) // 64))

    # ---- 3) 融合 + 归一化 ---------------------------------------------
    # HC 项平方锐化：大面积主导色也有对比度（背景 vs 别的颜色），平方把
    # 中低值压下去（背景掉到阈值下），主体（高频对比）保持——平滑背景
    # 保持大块、主体细分的行为不变。
    hc_n = _robust01(hc) ** 2
    spec_n = _robust01(sal_spec)
    sal = 0.6 * hc_n + 0.4 * spec_n
    sal = _box_blur(sal, max(1, min(w, h) // 96))      # 轻聚块
    sal = np.clip(sal, 0.0, None)
    mx = float(sal.max())
    if mx > 0:
        sal = sal / mx
    return sal


class SaliencyMap:
    """显著性积分图：O(1) 算任意矩形区域的平均显著性（与 DetailMap 同用法）。

    显著性按图内 95 分位归一化（主体永远凸显，与图的整体对比度无关），
    energy() 返回放大到与梯度阈值同量级的值（×250）。
    """

    def __init__(self, master: Image.Image, max_dim: int = 256):
        sal = saliency_map(master, max_dim)
        # 按 95 分位归一化（带下限地板，平坦图不被过度放大）
        p95 = float(np.percentile(sal, 95))
        gain = 1.0 / max(p95, 0.05)
        sal = np.clip(sal * gain, 0.0, 1.5)
        self.hh, self.hw = sal.shape
        self.I = np.zeros((self.hh + 1, self.hw + 1))
        np.cumsum(np.cumsum(sal, axis=0), axis=1, out=self.I[1:, 1:])
        W, H = master.size
        self.sx = W / self.hw
        self.sy = H / self.hh

    def energy(self, box) -> float:
        """区域平均相对显著性 × 250（与梯度阈值同量级，供 detail_threshold 复用）。"""
        x0, y0, x1, y1 = box
        cx0 = int(np.clip(np.floor(x0 / self.sx), 0, self.hw - 1))
        cy0 = int(np.clip(np.floor(y0 / self.sy), 0, self.hh - 1))
        cx1 = int(np.clip(np.ceil(x1 / self.sx), 1, self.hw))
        cy1 = int(np.clip(np.floor(y1 / self.sy), 1, self.hh))
        s = self.I[cy1, cx1] - self.I[cy0, cx1] - self.I[cy1, cx0] + self.I[cy0, cx0]
        return float(s / max(1, (cx1 - cx0) * (cy1 - cy0))) * 250.0
