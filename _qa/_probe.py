"""临时探针：验证 make_mosaic 同种子确定性（fresh vs shared matcher）与流式一致性。"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from photomosaic.db import TileDB
from photomosaic.library import ingest
from photomosaic.render import make_mosaic
from photomosaic.match import Matcher


def main() -> None:
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(7)
    tdir = os.path.join(tmp, "tiles")
    os.makedirs(tdir)
    for i in range(6):
        Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(
            os.path.join(tdir, f"t{i}.png"))
    master = os.path.join(tmp, "m.png")
    a = np.zeros((96, 96, 3), np.uint8)
    a[:, :48] = (220, 40, 40)
    a[:, 48:] = (40, 40, 220)
    yy, xx = np.ogrid[:96, :96]
    a[((yy - 48) ** 2 + (xx - 48) ** 2) < 400] = (240, 230, 60)
    Image.fromarray(a).save(master)

    db = TileDB(os.path.join(tmp, "t.db"))
    ingest(db, tdir, workers=0)

    for style in ("grid", "hex"):
        o1 = os.path.join(tmp, f"{style}_a.png")
        o2 = os.path.join(tmp, f"{style}_b.png")
        s1 = make_mosaic(master, db, o1, cells_x=6, style=style, seed=42, top_k=3)
        s2 = make_mosaic(master, db, o2, cells_x=6, style=style, seed=42, top_k=3)
        p1 = np.asarray(Image.open(o1).convert("RGB"))
        p2 = np.asarray(Image.open(o2).convert("RGB"))
        print(style, "fresh-matcher 同种子逐像素一致:", np.array_equal(p1, p2),
              "size:", s1["size"])

    # shared matcher（BatchRunner 模式）与 fresh 是否一致
    tiles = db.load_arrays()
    matcher = Matcher(tiles, device="cpu")
    o3 = os.path.join(tmp, "hex_shared.png")
    make_mosaic(master, db, o3, cells_x=6, style="hex", seed=42, top_k=3,
                shared_matcher=matcher)
    matcher.release()
    p3 = np.asarray(Image.open(o3).convert("RGB"))
    p_ref = np.asarray(Image.open(os.path.join(tmp, "hex_a.png")).convert("RGB"))
    print("hex shared-matcher vs fresh 逐像素一致:", np.array_equal(p3, p_ref))

    # 流式 vs 整图（grid / hex），多条件组合
    for style in ("grid", "hex", "dot"):
        full = os.path.join(tmp, f"{style}_full.png")
        stream = os.path.join(tmp, f"{style}_stream.png")
        make_mosaic(master, db, full, cells_x=6, style=style, seed=42, top_k=3,
                    stream_threshold_px=10 ** 12)
        make_mosaic(master, db, stream, cells_x=6, style=style, seed=42, top_k=3,
                    stream_threshold_px=1, stream_strip_rows=48)
        pf = np.asarray(Image.open(full).convert("RGB"))
        ps = np.asarray(Image.open(stream).convert("RGB"))
        same = np.array_equal(pf, ps)
        maxd = int(np.abs(pf.astype(int) - ps.astype(int)).max()) if not same else 0
        print(f"{style}: 流式 vs 整图 逐像素一致={same} max_diff={maxd}")

    db.close()
    print("PROBE OK")


if __name__ == "__main__":
    main()
