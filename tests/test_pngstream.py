"""流式 PNG 编码器：IHDR+zlib 逐行+IDAT 块，与 PIL 读回逐像素一致。"""

import numpy as np
import pytest
from PIL import Image

from photomosaic.pngstream import PngStreamWriter


def _pattern(w, h):
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    a = np.empty((h, w, 3), dtype=np.uint8)
    a[..., 0] = (yy + xx) % 256
    a[..., 1] = (yy * 3) % 256
    a[..., 2] = (xx * 5) % 256
    return a


def test_png_roundtrip_multistrip(tmp_path):
    W, H, RPS = 97, 250, 64
    exp = _pattern(W, H)
    out = tmp_path / "t.png"
    with PngStreamWriter(str(out), W, H) as w:
        for si in range((H + RPS - 1) // RPS):
            y0, y1 = si * RPS, min(H, (si + 1) * RPS)
            w.write_strip(si, exp[y0:y1])
    got = np.asarray(Image.open(out).convert("RGB"))
    assert np.array_equal(got, exp)


def test_png_single_strip(tmp_path):
    W = H = 32
    exp = _pattern(W, H)
    out = tmp_path / "one.png"
    with PngStreamWriter(str(out), W, H) as w:
        w.write_strip(0, exp)
    assert np.array_equal(np.asarray(Image.open(out).convert("RGB")), exp)


def test_png_rejects_out_of_order(tmp_path):
    exp = _pattern(8, 8)
    out = tmp_path / "bad.png"
    with pytest.raises(ValueError, match="顺序"):
        with PngStreamWriter(str(out), 8, 8) as w:
            w.write_strip(1, exp[0:4])


def test_png_rejects_wrong_width(tmp_path):
    out = tmp_path / "w.png"
    with pytest.raises(ValueError, match="宽度"):
        with PngStreamWriter(str(out), 8, 8) as w:
            w.write_strip(0, np.zeros((8, 4, 3), dtype=np.uint8))
