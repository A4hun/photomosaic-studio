# -*- coding: utf-8 -*-
"""QA-A：批量出图全流程（真实离屏 GUI 驱动）。

真实用户路径：
  ① 队列加 3 张主图 → ② 点"构建数据库" → ③ 设配置（风格=六边形蜂窝、
     基础块数=8、细节识别=显著性、固定种子）→ 其中一张用"单独配置"对话框
     改成圆点点彩 + 基础块数 10 + 每格像素 40 → ④ 点"生成"（队列>1 自动走批量）。

断言：
  * BatchRunner 收到的 base_params 与③页配置逐项一致（间谍捕获）；
  * 产出 3 张成品，尺寸跟随配置（两张与主图同大、覆盖那张 400×300）；
  * 每张与"同参数直接调 make_mosaic"的参照渲染逐像素一致（真跟随③配置）；
  * 蜂窝成品在格角处露出高斯模糊主图背景（有缝隙背景）；
  * 覆盖风格的那张内容与统一风格参照不同（覆盖生效）；
  * 命名含风格后缀且后缀 = 实际渲染风格（单独配置改风格的也要对）；
  * 批量中断（should_cancel）后重跑：已完成跳过、剩余补齐。

运行：.venv\\Scripts\\python _qa\\qa_batch_render.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
from PIL import Image

import common  # _qa 共享助手（离屏/弹窗打桩/worker 等待）

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "out", "qa_batch")


def _patch_override_dialog(style_label: str, cells_x: int, cell_px: int):
    """把「单独配置」QDialog.exec 打桩成：选好风格/块数/每格像素后接受。"""
    from PySide6.QtWidgets import QComboBox, QDialog, QSpinBox

    saved = QDialog.exec

    def fake_exec(dlg):
        try:
            title = dlg.windowTitle()
        except Exception:
            title = ""
        if title.startswith("单独配置"):
            combos = dlg.findChildren(QComboBox)
            spins = dlg.findChildren(QSpinBox)
            idx = combos[0].findText(style_label)
            assert idx > 0, f"风格下拉找不到 {style_label}"
            combos[0].setCurrentIndex(idx)
            # 创建顺序：基础块数、细分层级、融合强度、每格像素
            spins[0].setValue(cells_x)
            spins[3].setValue(cell_px)
            print(f"  [对话框] 单独配置: 风格={style_label} 块数={cells_x} "
                  f"每格像素={cell_px} → 接受")
            return QDialog.DialogCode.Accepted
        return QDialog.DialogCode.Rejected

    QDialog.exec = fake_exec
    return saved


def _rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def main() -> int:
    from photomosaic.db import TileDB
    from photomosaic.render import make_mosaic

    ck = common.Checker("QA-A 批量出图")
    common.get_app()
    common.patch_dialogs()

    # ---------------- 工作区 ----------------
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT)
    tiles_dir = os.path.join(OUT_ROOT, "tiles")
    common.make_tiles(tiles_dir, n=12)
    masters = []
    for i in range(3):
        p = os.path.join(OUT_ROOT, f"m{i}.png")
        common.make_master(p, 240, 180, seed=3 + i)
        masters.append(p)
    db_path = os.path.join(OUT_ROOT, "tiles.db")
    out1 = os.path.join(OUT_ROOT, "out1")
    out2 = os.path.join(OUT_ROOT, "out2")
    os.makedirs(out1)
    os.makedirs(out2)

    from photomosaic.gui import MainWindow

    win = MainWindow(load_settings=False)

    # ---------------- ① 队列加 3 张主图 ----------------
    ck.section("① 主图队列")
    for p in masters:
        win.master_queue.addItem(p)
    win.master_queue.setCurrentRow(0)          # 触发 _on_master_queue_select
    common.pump()
    ck.check(win.master_queue.count() == 3, "队列有 3 张主图")
    ck.check(win.master_path == masters[0], "选中第 1 张为当前主图")
    ck.check(win.tree is not None, "主图加载后网格树已建")

    # ---------------- ② 点按钮建库 ----------------
    ck.section("② 构建单元图数据库（点按钮，走 _Worker）")
    win.folder_list.addItem(tiles_dir)
    win.db_edit.setText(db_path)
    win.btn_ingest.click()
    ck.check(common.wait_worker(win, 300), "建库 worker 完成")
    common.pump(50)
    with TileDB(db_path) as db:
        n_tiles = db.count()
    ck.check(n_tiles == 24, "库行数 = 12 张 × (原图+镜像)", f"实际 {n_tiles}")
    ck.check("完成" in win.status_text.text(), "状态栏提示构建完成",
             win.status_text.text())

    # ---------------- ③ 设置渲染配置 ----------------
    ck.section("③ 网格与风格配置")
    win._style_buttons["hex"].click()          # 六边形蜂窝
    win.cells_x_spin.setValue(8)
    win.signal_combo.setCurrentIndex(1)        # 显著性主体
    win.level_radios[1].setChecked(True)       # 二级
    win.auto_check.setChecked(True)
    win.seed_spin.setValue(42)                 # 固定种子 → 可复现
    win.outdir_edit.setText(out1)
    common.pump(50)
    ck.check(win._style == "hex", "风格=hex")

    # 其中一张（m1）设单独配置：圆点点彩 + 块数 10 + 每格像素 40
    saved_exec = _patch_override_dialog("圆点点彩", 10, 40)
    try:
        win.master_queue.setCurrentRow(1)
        common.pump()
        win._configure_master()                # 真实走对话框逻辑（exec 已打桩）
    finally:
        from PySide6.QtWidgets import QDialog
        QDialog.exec = saved_exec
    win.master_queue.setCurrentRow(0)
    common.pump()
    ov = win._master_overrides.get(masters[1])
    ck.check(ov == {"style": "dot", "cells_x": 10, "cell_px": 40},
             "m1 单独配置已写入", repr(ov))
    ck.check(win.master_queue.item(1).font().bold(), "m1 队列项加粗标记")

    # ---------------- BatchRunner 间谍 ----------------
    import photomosaic.batch as batch_mod

    run_calls: list[dict] = []
    OrigRunner = batch_mod.BatchRunner

    class SpyRunner(OrigRunner):
        def run(self, masters_, out_dir_, **kw):
            stats = super().run(masters_, out_dir_, **kw)
            run_calls.append({"masters": list(masters_), "out_dir": out_dir_,
                              "kw": dict(kw), "stats": stats})
            return stats

    batch_mod.BatchRunner = SpyRunner

    # ---------------- ④ 点"生成"（应自动走批量） ----------------
    ck.section("④ 生成（队列>1 → 批量）")
    win.btn_render.click()
    ck.check(common.wait_worker(win, 600), "批量 worker 完成")
    common.pump(50)
    batch_mod.BatchRunner = OrigRunner

    ck.check(len(run_calls) == 1, "BatchRunner.run 被调用 1 次（自动批量）",
             f"实际 {len(run_calls)}")
    call = run_calls[0]
    kw = call["kw"]
    stats = call["stats"]
    bp = kw.get("base_params") or {}
    expect_bp = {"cells_x": 8, "levels": 2, "detail_threshold": 25.0,
                 "top_k": 8, "max_reuse": 0, "seed": 42, "blend": 0.55,
                 "style": "hex", "signal": "saliency", "first_level_factor": 1.0}
    for k_, v_ in expect_bp.items():
        ck.check(bp.get(k_) == v_, f"base_params.{k_} == {v_!r}",
                 f"实际 {bp.get(k_)!r}")
    ck.check(kw.get("styles") == ["hex"], "批量风格矩阵 = [hex]",
             repr(kw.get("styles")))
    ck.check(call["masters"] == masters, "批量主图 = 队列 3 张")
    ck.check(kw.get("overrides") == {masters[1]: {"style": "dot", "cells_x": 10,
                                                  "out_width": 400}},
             "m1 覆盖参数（cell_px 已换算 out_width=400）", repr(kw.get("overrides")))
    ck.check(stats.get("ok") == 3 and stats.get("failed") == 0,
             "批量统计 ok=3 failed=0", repr(stats))

    # ---------------- 产出断言 ----------------
    ck.section("产出文件")
    names = sorted(os.listdir(out1))
    print("  out1:", names)
    p0 = os.path.join(out1, "m0_hex_mosaic.png")
    p2 = os.path.join(out1, "m2_hex_mosaic.png")
    ck.check(os.path.exists(p0), "m0 产出存在且带 _hex 后缀")
    ck.check(os.path.exists(p2), "m2 产出存在且带 _hex 后缀")
    # m1 覆盖成 dot：文件名后缀应反映实际风格
    p1_dot = os.path.join(out1, "m1_dot_mosaic.png")
    p1_hex = os.path.join(out1, "m1_hex_mosaic.png")
    ck.check(os.path.exists(p1_dot) and not os.path.exists(p1_hex),
             "m1 产出文件名后缀 = 实际渲染风格 dot",
             f"实际: hex={os.path.exists(p1_hex)} dot={os.path.exists(p1_dot)}")
    p1 = p1_dot if os.path.exists(p1_dot) else p1_hex

    imgs = {p: _rgb(p) for p in (p0, p1, p2) if os.path.exists(p)}
    if p0 in imgs:
        ck.check(imgs[p0].shape[:2] == (180, 240),
                 "m0 输出尺寸 = 主图尺寸 240×180", repr(imgs[p0].shape))
    if p2 in imgs:
        ck.check(imgs[p2].shape[:2] == (180, 240),
                 "m2 输出尺寸 = 主图尺寸 240×180", repr(imgs[p2].shape))
    if p1 in imgs:
        ck.check(imgs[p1].shape[:2] == (300, 400),
                 "m1 输出尺寸 = 覆盖配置 400×300", repr(imgs[p1].shape))

    # 与"同参数直接渲染"逐像素一致（证明批量真的跟随③配置）
    ck.section("与参照渲染逐像素对比")
    ref_params = dict(bp)
    with TileDB(db_path) as db:
        for mpath, outp, extra in (
                (masters[0], p0, {"style": "hex"}),
                (masters[2], p2, {"style": "hex"}),
                (masters[1], p1, {"style": "dot", "cells_x": 10,
                                  "out_width": 400})):
            if not os.path.exists(outp):
                continue
            rp = dict(ref_params)
            rp.update(extra)
            ref = os.path.join(OUT_ROOT, f"ref_{os.path.basename(outp)}")
            make_mosaic(mpath, db, ref, **rp)
            same = _rgb(ref).shape == imgs[outp].shape \
                and np.array_equal(_rgb(ref), imgs[outp])
            ck.check(same, f"{os.path.basename(outp)} 与参照渲染逐像素一致")

    # 蜂窝缝隙背景：格角像素 ≈ 高斯模糊的主图底色（低饱和灰），不是瓦片色
    if p0 in imgs:
        corner = imgs[p0][2, 2].astype(int)
        ck.check(int(corner.max() - corner.min()) < 40,
                 f"hex 格角露模糊背景（近似灰底）rgb={corner.tolist()}")
    # 覆盖生效：m1 内容与 hex 参照不同
    if p1 in imgs:
        ref_m1_hex = os.path.join(OUT_ROOT, "ref_m1hex.png")
        with TileDB(db_path) as db:
            rp = dict(ref_params)
            rp["style"] = "hex"
            make_mosaic(masters[1], db, ref_m1_hex, **rp)
        m1hex = _rgb(ref_m1_hex)
        ck.check(not (m1hex.shape == imgs[p1].shape
                      and np.array_equal(m1hex, imgs[p1])),
                 "m1 单独配置(dot) 与统一配置(hex) 产出不同")
    ck.check("批量完成 3/3" in win.status_text.text(), "状态栏：批量完成 3/3",
             win.status_text.text())

    # ---------------- 批量中断 → 重跑跳过已完成 ----------------
    ck.section("批量中断 / 断点续批")
    win.outdir_edit.setText(out2)
    run_calls.clear()

    class CancelAfterFirst(SpyRunner):
        def run(self, masters_, out_dir_, **kw2):
            orig_progress = kw2.get("progress")

            def wrapped(d, t, p):
                if orig_progress:
                    orig_progress(d, t, p)
                if d >= 1:
                    win._render_cancel.set()     # 第 1 张完成后置取消（worker 线程内）

            kw2["progress"] = wrapped
            return super().run(masters_, out_dir_, **kw2)

    batch_mod.BatchRunner = CancelAfterFirst
    win.btn_render.click()
    ck.check(common.wait_worker(win, 600), "中断批量 worker 完成")
    common.pump(50)
    st1 = run_calls[-1]["stats"]
    names2 = sorted(os.listdir(out2))
    print("  out2 (中断后):", names2, "stats:", st1)
    ck.check(len(names2) == 1 and names2[0].startswith("m0_"),
             "中断后只产出第 1 张", repr(names2))
    ck.check(st1.get("ok") == 1 and st1.get("done") == 1,
             "中断统计 ok=1 done=1", repr(st1))

    # 重跑：已存在 → 跳过；其余补齐
    batch_mod.BatchRunner = SpyRunner
    win.btn_render.click()
    ck.check(common.wait_worker(win, 600), "续跑批量 worker 完成")
    common.pump(50)
    batch_mod.BatchRunner = OrigRunner
    st2 = run_calls[-1]["stats"]
    names3 = sorted(os.listdir(out2))
    print("  out2 (续跑后):", names3, "stats:", st2)
    ck.check(len(names3) == 3, "续跑后 3 张齐", repr(names3))
    ck.check(st2.get("skipped") == 1 and st2.get("ok") == 2,
             "续跑统计 skipped=1 ok=2", repr(st2))
    ck.check("跳过 1" in win.status_text.text(),
             "状态栏显示跳过 1（已完成不重渲）", win.status_text.text())

    win.close()
    common.pump(50)
    ck.check(not [c for c in common.MSGBOX_CALLS if c[0] in ("warning", "critical")],
             "全程无 warning/critical 弹窗", repr(common.MSGBOX_CALLS[:3]))
    return ck.summary()


if __name__ == "__main__":
    sys.exit(common.run_guarded(main))
