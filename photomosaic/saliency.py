"""显著性检测（saliency detection）：找出图像的视觉重点/主体区域。

实现 Hou & Zhang 2007 频谱残差法（Spectral Residual），纯 numpy/FFT，
零新依赖、离线可用。比"梯度能量"更懂内容：天空渐变/平滑背景不会被误判
为重点，人脸/主体/高对比主体会被凸显。

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


def saliency_map(master: Image.Image, max_dim: int = 256) -> np.ndarray:
    """返回 (h, w) 的 0-1 显著性图（降采样到 max_dim 内，保持比例）。"""
    W, H = master.size
    f = max(1, int(max(W, H) / max_dim))
    w, h = max(2, W // f), max(2, H // f)
    gray = np.asarray(
        master.convert("L").resize((w, h), Image.Resampling.BOX),
        dtype=np.float64)

    # 频谱残差：log 幅度谱 - 其局部均值（低通），再拼回相位做逆变换
    spec = np.fft.fft2(gray)
    amp = np.abs(spec)
    phase = np.angle(spec)
    log_amp = np.log(amp + 1e-9)
    mean = _box_blur(log_amp, 2)
    residual = log_amp - mean

    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    # 高斯式平滑（盒式模糊近似）+ 归一化到 0-1
    sal = _box_blur(sal, max(1, min(w, h) // 64))
    sal -= sal.min()
    if sal.max() > 0:
        sal /= sal.max()
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
