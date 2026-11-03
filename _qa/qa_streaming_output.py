# -*- coding: utf-8 -*-
"""QA-C：超大流式输出（强制流式，BigTIFF + PNG + 断点续传）。

make_mosaic(stream_threshold_px=1) 强制走分条带流式路径，断言：
  * BigTIFF 流式成品与整图路径（同参同种子）逐像素一致（tifffile 读回）；
  * PNG 流式成品与整图路径逐像素一致（PIL 读回）；
  * uniform 与 adaptive（levels=2）两条路径都成立；
  * BigTIFF 渲染一半取消 → meta.json 记 done_strips → 重跑从第 4 条带续
    （间谍验证 done_strips=3 传入）→ 结果与一次跑完逐像素一致；
  * PNG 不可续传 → 重跑从头写出 → 结果仍与一次跑完一致；
  * 成功后 .ckpt 目录被清理；
  * 【已知差异·上报】蜂窝(hex)/点彩(dot)这类"有缝隙"风格的流式背景是
    按条带分别高斯模糊的，与整图路径不逐像素一致（格缝处 max diff 大），
    此处只量化上报，不作为通过条件（修复方案需 Lead 决策）。

运行：.venv\\Scripts\\python _qa\\qa_streaming_output.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

import common

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "out", "qa_stream")


def read_pixels(path: str) -> np.ndarray:
    """读图到 RGB ndarray；TIFF/BigTIFF 走 tifffile，其余走 PIL。"""
    if path.lower().endswith((".tif", ".tiff")):
        try:
            import tifffile
            arr = tifffile.imread(path)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return arr[..., :3].astype(np.uint8)
            return np.repeat(arr[..., None], 3, axis=2).astype(np.uint8)
        except Exception:
            pass
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _diff(a: np.ndarray, b: np.ndarray) -> tuple[bool, int]:
    if a.shape != b.shape:
        return False, -1
    if np.array_equal(a, b):
        return True, 0
    return False, int(np.abs(a.astype(int) - b.astype(int)).max())


def main() -> int:
    from photomosaic.db import TileDB
    from photomosaic.library import ingest
    from photomosaic.render import RenderCancelled, make_mosaic

    ck = common.Checker("QA-C 超大流式输出")
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT)
    tiles_dir = os.path.join(OUT_ROOT, "tiles")
    common.make_tiles(tiles_dir, n=12)
    master = os.path.join(OUT_ROOT, "master.png")
    common.make_master(master, 160, 120, seed=5)
    db_path = os.path.join(OUT_ROOT, "tiles.db")
    with TileDB(db_path) as _db:
        ingest(_db, tiles_dir)

    db = TileDB(db_path)
    base = dict(cells_x=5, seed=42, top_k=3)      # uniform: 320×256 输出
    STRIPS = dict(stream_threshold_px=1, stream_strip_rows=40)   # 7 条带
    FULL = dict(stream_threshold_px=10 ** 12)

    # ---------- 1. BigTIFF / PNG 流式 vs 整图 逐像素一致 ----------
    ck.section("流式 vs 整图 逐像素一致（grid，uniform）")
    ref_tif = os.path.join(OUT_ROOT, "full.tif")
    st_tif = os.path.join(OUT_ROOT, "stream.tif")
    make_mosaic(master, db, ref_tif, **base, **FULL)
    make_mosaic(master, db, st_tif, **base, **STRIPS)
    a, b = read_pixels(ref_tif), read_pixels(st_tif)
    same, md = _diff(a, b)
    ck.check(a.shape == (256, 320, 3), "输出尺寸 320×256", repr(a.shape))
    ck.check(same, f"BigTIFF 流式 == 整图路径（max_diff={md}）")

    ref_png = os.path.join(OUT_ROOT, "full.png")
    st_png = os.path.join(OUT_ROOT, "stream.png")
    make_mosaic(master, db, ref_png, **base, **FULL)
    make_mosaic(master, db, st_png, **base, **STRIPS)
    same, md = _diff(read_pixels(ref_png), read_pixels(st_png))
    ck.check(same, f"PNG 流式 == 整图路径（max_diff={md}）")

    # adaptive（levels=2 + 显著性）路径的流式一致
    ck.section("流式 vs 整图（adaptive: levels=2 + saliency）")
    ad = dict(base, levels=2, signal="saliency", detail_threshold=30.0)
    ref_a = os.path.join(OUT_ROOT, "full_ad.png")
    st_a = os.path.join(OUT_ROOT, "stream_ad.png")
    make_mosaic(master, db, ref_a, **ad, **FULL)
    make_mosaic(master, db, st_a, **ad, **STRIPS)
    same, md = _diff(read_pixels(ref_a), read_pixels(st_a))
    ck.check(same, f"adaptive PNG 流式 == 整图（max_diff={md}）")

    # ---------- 2. BigTIFF 断点续传 ----------
    ck.section("BigTIFF 断点续传：半取消 → 重跑 → 与一次跑完一致")
    one_shot = os.path.join(OUT_ROOT, "bt_oneshot.tif")
    resumed = os.path.join(OUT_ROOT, "bt_resume.tif")
    make_mosaic(master, db, one_shot, **base, **STRIPS)

    flag = {"cancel": False}
    armed = {"on": False}

    def status_arm(msg):     # 匹配阶段的进度会先走 progress；只在条带阶段武装
        if "流式渲染" in str(msg):
            armed["on"] = True

    def progress_cut(done, total):
        if armed["on"] and done >= 3:
            flag["cancel"] = True

    try:
        make_mosaic(master, db, resumed, **base, **STRIPS,
                    should_cancel=lambda: flag["cancel"],
                    progress=progress_cut, status=status_arm)
        ck.check(False, "渲染一半应抛出 RenderCancelled")
    except RenderCancelled:
        ck.check(True, "渲染到一半被取消（RenderCancelled）")
    ck.check(os.path.exists(resumed), "取消后已写出部分 BigTIFF 文件")
    meta_p = os.path.join(resumed + ".ckpt", "meta.json")
    ck.check(os.path.exists(meta_p), "断点 meta.json 已保存")
    if os.path.exists(meta_p):
        with open(meta_p, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        ck.check(meta.get("done_strips") == 3, "断点记录 done_strips=3",
                 repr(meta))

    # 间谍：确认重跑真的传入 done_strips=3（而不是从头重渲）
    import photomosaic.streaming as streaming_mod

    seen: list[int] = []
    orig_rs = streaming_mod.render_streaming

    def spy_rs(*a, **kw):
        seen.append(int(kw.get("done_strips", 0)))
        return orig_rs(*a, **kw)

    streaming_mod.render_streaming = spy_rs
    try:
        stats2 = make_mosaic(master, db, resumed, **base, **STRIPS)
    finally:
        streaming_mod.render_streaming = orig_rs
    ck.check(seen == [3], "重跑从第 4 条带续传（done_strips=3）", repr(seen))
    same, md = _diff(read_pixels(one_shot), read_pixels(resumed))
    ck.check(same, f"续传成品 == 一次跑完（max_diff={md}）")
    ck.check(stats2.get("streamed") is True and stats2.get("strips") == 7,
             "续传统计 streamed/strips=7",
             repr({k: stats2.get(k) for k in ("streamed", "strips")}))
    ck.check(not os.path.exists(resumed + ".ckpt"), "成功后 .ckpt 已清理")

    # ---------- 3. PNG 取消 → 重跑（不可续传，从头写） ----------
    ck.section("PNG 取消 → 重跑（从头写，结果一致）")
    png_one = os.path.join(OUT_ROOT, "pg_oneshot.png")
    png_re = os.path.join(OUT_ROOT, "pg_resume.png")
    make_mosaic(master, db, png_one, **base, **STRIPS)
    flag["cancel"] = False
    armed["on"] = False
    try:
        make_mosaic(master, db, png_re, **base, **STRIPS,
                    should_cancel=lambda: flag["cancel"],
                    progress=progress_cut, status=status_arm)
        ck.check(False, "PNG 渲染一半应抛出 RenderCancelled")
    except RenderCancelled:
        ck.check(True, "PNG 渲染到一半被取消")
    stats3 = make_mosaic(master, db, png_re, **base, **STRIPS)
    same, md = _diff(read_pixels(png_one), read_pixels(png_re))
    ck.check(same, f"PNG 重跑成品 == 一次跑完（max_diff={md}）")
    ck.check(stats3.get("streamed") is True, "PNG 重跑仍走流式")
    ck.check(not os.path.exists(png_re + ".ckpt"), "PNG 成功后 .ckpt 已清理")

    # ---------- 4. 有缝隙风格（hex/dot）：流式与整图的背景差异量化上报 ----------
    ck.section("有缝隙风格：流式 vs 整图 背景差异（上报项）")
    for style in ("hex", "dot"):
        f = os.path.join(OUT_ROOT, f"{style}_full.png")
        s = os.path.join(OUT_ROOT, f"{style}_stream.png")
        make_mosaic(master, db, f, **base, **FULL, style=style)
        make_mosaic(master, db, s, **base, **STRIPS, style=style)
        same, md = _diff(read_pixels(f), read_pixels(s))
        if same:
            ck.check(True, f"{style} 流式 == 整图")
        else:
            print(f"  [已知差异] {style}: 流式与整图不逐像素一致, "
                  f"max_diff={md}（条带独立高斯模糊背景所致，详见 QA 汇报 bug#2）")

    db.close()
    return ck.summary()


if __name__ == "__main__":
    sys.exit(common.run_guarded(main))
