import numpy as np
from PIL import Image

from photomosaic.features import (
    FEATURE_DIM,
    average_rgb,
    crop_to_ratio,
    decode_vector,
    encode_vector,
    subblock_lab,
)


def test_solid_colour_feature_is_uniform():
    img = Image.new("RGB", (90, 90), (200, 30, 30))
    feat = subblock_lab(img)
    assert feat.shape == (FEATURE_DIM,)
    # layout is interleaved per sub-block: (L,a,b) x 9
    L, a, b = feat[0::3], feat[1::3], feat[2::3]
    assert np.allclose(L, L[0]) and np.allclose(a, a[0]) and np.allclose(b, b[0])
    # red: L in (0,100), a strongly positive
    assert 0 < L[0] < 100
    assert a[0] > 20


def test_average_rgb():
    img = Image.new("RGB", (10, 10), (10, 200, 50))
    avg = average_rgb(img)
    assert avg.shape == (3,)
    assert np.allclose(avg, [10, 200, 50], atol=1)


def test_crop_to_ratio_landscape_and_portrait():
    img = Image.new("RGB", (200, 100))
    assert crop_to_ratio(img, 1.0).size == (100, 100)
    assert crop_to_ratio(img, 2.0).size == (200, 100)
    tall = Image.new("RGB", (50, 200))
    assert crop_to_ratio(tall, 0.5).size == (50, 100)


def test_vector_roundtrip():
    vec = np.arange(FEATURE_DIM, dtype=np.float32) * 1.5
    assert np.array_equal(decode_vector(encode_vector(vec)), vec)


def test_features_distinguish_structures():
    # same average colour, different 3x3 structure -> features must differ
    a = Image.new("RGB", (90, 90), (128, 128, 128))
    b = Image.new("RGB", (90, 90), (128, 128, 128))
    for x in range(0, 90, 30):  # checkerboard-ish blocks
        for y in range(0, 90, 30):
            if (x // 30 + y // 30) % 2 == 0:
                b.paste((255, 255, 255), (x, y, x + 30, y + 30))
            else:
                b.paste((0, 0, 0), (x, y, x + 30, y + 30))
    fa, fb = subblock_lab(a), subblock_lab(b)
    assert np.allclose(fa, fb) is False
    assert float(((fa - fb) ** 2).sum()) > 100
