"""Feature extraction.

Each image (tile or master cell) is described by a 27-dimensional vector:
the image is box-downsampled to a 3x3 grid of sub-block averages, converted
to CIE LAB, and flattened to 9 L values + 9 a values + 9 b values. Matching
compares these vectors with squared Euclidean distance, which captures local
colour structure instead of a single global average.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

FEATURE_DIM = 27  # 3x3 sub-blocks x 3 LAB channels

# Pillow "LAB" mode (verified empirically on Pillow 13): L is stored as
# 0..255 (L* x 255/100); a and b are stored as signed int8 values in uint8
# containers (two's-complement wrap, neutral = 0, e.g. -64 -> 192).
_L_SCALE = 100.0 / 255.0


_LAB_TRANSFORM = None


def _to_lab(img: Image.Image) -> Image.Image:
    """RGB -> Pillow LAB, bit-identical to ``img.convert("LAB")``.

    Pillow rebuilds the sRGB->LAB colour transform on *every* convert("LAB")
    call (~10 ms), which dominated per-tile cost; build it once per process.
    """
    global _LAB_TRANSFORM
    if img.mode != "RGB":
        img = img.convert("RGB")
    try:
        from PIL import ImageCms
        if _LAB_TRANSFORM is None:
            _LAB_TRANSFORM = ImageCms.buildTransform(
                ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"),
                "RGB", "LAB")
        return ImageCms.applyTransform(img, _LAB_TRANSFORM)
    except Exception:                      # no littlecms etc. -> original path
        return img.convert("LAB")


def lab_array(img: Image.Image) -> np.ndarray:
    """Convert an RGB image to a float32 LAB array with true LAB ranges."""
    arr = np.asarray(_to_lab(img))
    out = np.empty(arr.shape, dtype=np.float32)
    out[..., 0] = arr[..., 0].astype(np.float32) * _L_SCALE
    out[..., 1] = arr[..., 1].astype(np.int8).astype(np.float32)
    out[..., 2] = arr[..., 2].astype(np.int8).astype(np.float32)
    return out


def subblock_lab(img: Image.Image, blocks: int = 3) -> np.ndarray:
    """Return the (blocks*blocks*3,) LAB sub-block feature vector."""
    small = img.convert("RGB").resize((blocks, blocks), Image.Resampling.BOX)
    return lab_array(small).reshape(-1).astype(np.float32)


def average_rgb(img: Image.Image) -> np.ndarray:
    """Return the (3,) float32 average RGB (0..255) used for radius filtering."""
    px = img.convert("RGB").resize((1, 1), Image.Resampling.BOX)
    return np.asarray(px, dtype=np.float32).reshape(3)


def crop_to_ratio(img: Image.Image, ratio: float) -> Image.Image:
    """Centre-crop to the given width/height ratio without scaling."""
    w, h = img.size
    current = w / h
    if abs(current - ratio) < 1e-6:
        return img
    if current > ratio:
        new_w = max(1, int(round(h * ratio)))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = max(1, int(round(w / ratio)))
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def encode_vector(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
