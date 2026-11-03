# -*- coding: utf-8 -*-
"""QA-B：设置记忆全流程（真实离屏 GUI，关窗→重开逐项恢复）。

真实用户路径：
  开窗口 → 改一大批设置（阈值/风格/融合/细节识别/主图队列/单独配置/
  默认保存到/缩放/导出格式/导出质量/每格像素/建库项/匹配项/当前页/旋转）
  → 关窗（closeEvent 写 SETTINGS_PATH）→ 再开新窗口（构造时读回）
  → 逐项断言恢复。

安全防护（硬约束）：
  * photomosaic.gui.SETTINGS_PATH 打桩到临时文件——绝不碰真实
    ~/.photomosaic_studio.json；脚本结束校验真实文件未被创建/修改。
  * 另做一条 load_settings=False + _save_state/_apply_state 到临时 json
    的纯数据回路（任务要求的双通道验证）。

运行：.venv\\Scripts\\python _qa\\qa_settings_memory.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import common

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "out", "qa_settings")
REAL_SETTINGS = os.path.join(os.path.expanduser("~"),
                             ".photomosaic_studio.json")


def _real_settings_snapshot():
    if not os.path.exists(REAL_SETTINGS):
        return None
    st = os.stat(REAL_SETTINGS)
    return (st.st_mtime_ns, st.st_size)


def _set_all(win, ctx) -> dict:
    """在窗口上改一大批设置；返回期望值字典（供逐项断言）。"""
    # --- ③ 网格/匹配/风格 ---
    win.threshold_slider.setValue(61)
    win._style_buttons["dot"].click()          # 圆点点彩
    win.blend_slider.setValue(77)
    win.signal_combo.setCurrentIndex(1)        # 显著性
    win.cells_x_spin.setValue(133)
    win.level_radios[1].setChecked(True)       # 二级
    win.auto_check.setChecked(False)           # 关掉自动细分（也要记住）
    win.topk_spin.setValue(5)
    win.reuse_spin.setValue(12)
    win.radius_spin.setValue(3.5)
    win.seed_spin.setValue(7)
    win.device_combo.setCurrentText("cpu")
    win.first_level_check.setChecked(True)
    win.skip_white_check.setChecked(True)
    win.skip_black_check.setChecked(True)
    # --- ② 建库 ---
    win.db_edit.setText(ctx["db"])
    win.folder_list.addItem(ctx["folder1"])
    win.folder_list.addItem(ctx["folder2"])
    win.variant_checks["orig"].setChecked(True)
    win.variant_checks["mirror"].setChecked(False)
    win.variant_checks["r90"].setChecked(True)
    win.variant_checks["r180"].setChecked(False)
    win.variant_checks["r270"].setChecked(False)
    win.thumb_spin.setValue(128)
    win.dedup_check.setChecked(False)
    win.subfolder_check.setChecked(False)
    win.cell_quality_spin.setValue(85)
    win.workers_spin.setValue(2)
    # --- ① 主图队列 + 单独配置 ---
    for p in ctx["masters"]:
        win.master_queue.addItem(p)
    win.master_queue.setCurrentRow(0)
    win._master_overrides[ctx["masters"][1]] = {"style": "dot", "cells_x": 20}
    it = win.master_queue.item(1)
    f = it.font()
    f.setBold(True)
    it.setFont(f)
    it.setToolTip("单独配置: style=dot, cells_x=20")
    # --- ④ 结果页 ---
    win.outdir_edit.setText(ctx["outdir"])
    win.zoom_combo.setCurrentText("100%")
    win.format_combo.setCurrentText("JPEG")
    win.out_quality.setValue(90)
    win.cell_px_spin.setValue(24)
    # --- 当前页 / 旋转 / 欢迎标记 ---
    win._goto(2)
    win.master_rot = 90                        # _set_current_master 之后改
    win._welcome_shown = True
    return {
        "threshold": 61, "style": "dot", "blend": 77, "signal": 1,
        "cells_x": 133, "level": 2, "auto": False, "top_k": 5,
        "max_reuse": 12, "radius": 3.5, "seed": 7, "device": "cpu",
        "first_level": True, "skip_white": True, "skip_black": True,
        "db": ctx["db"], "folders": [ctx["folder1"], ctx["folder2"]],
        "variants": {"orig": True, "mirror": False, "r90": True,
                     "r180": False, "r270": False},
        "thumb": 128, "dedup": False, "subfolders": False,
        "cell_quality": 85, "workers": 2,
        "masters": ctx["masters"],
        "overrides": {ctx["masters"][1]: {"style": "dot", "cells_x": 20}},
        "out_dir": ctx["outdir"], "zoom": "100%", "format": "JPEG",
        "out_quality": 90, "cell_px": 24, "page": 2,
        "master": ctx["masters"][0], "master_rot": 90,
        "welcome_shown": True,
    }


def _assert_all(ck, win, exp: dict, tag: str) -> None:
    """逐项断言窗口状态与期望值一致。"""
    ck.check(win.threshold_slider.value() == exp["threshold"],
             f"{tag}阈值={exp['threshold']}", str(win.threshold_slider.value()))
    ck.check(win._style == exp["style"], f"{tag}风格={exp['style']}",
             win._style)
    ck.check(all(b.isChecked() == (n == exp["style"])
                 for n, b in win._style_buttons.items()),
             f"{tag}风格卡片选中态一致")
    ck.check(win.blend_slider.value() == exp["blend"],
             f"{tag}融合={exp['blend']}", str(win.blend_slider.value()))
    ck.check(win.signal_combo.currentIndex() == exp["signal"],
             f"{tag}细节识别=显著性", str(win.signal_combo.currentIndex()))
    ck.check(win.cells_x_spin.value() == exp["cells_x"],
             f"{tag}基础块数={exp['cells_x']}", str(win.cells_x_spin.value()))
    ck.check(win._radio_level() == exp["level"],
             f"{tag}层级={exp['level']}", str(win._radio_level()))
    ck.check(win.auto_check.isChecked() == exp["auto"],
             f"{tag}自动细分={exp['auto']}")
    ck.check(win.topk_spin.value() == exp["top_k"], f"{tag}top_k")
    ck.check(win.reuse_spin.value() == exp["max_reuse"], f"{tag}最大复用")
    ck.check(abs(win.radius_spin.value() - exp["radius"]) < 1e-9,
             f"{tag}颜色半径={exp['radius']}", str(win.radius_spin.value()))
    ck.check(win.seed_spin.value() == exp["seed"], f"{tag}随机种子")
    ck.check(win.device_combo.currentText() == exp["device"],
             f"{tag}设备={exp['device']}", win.device_combo.currentText())
    ck.check(win.first_level_check.isChecked() == exp["first_level"],
             f"{tag}首级保守")
    ck.check(win.skip_white_check.isChecked() == exp["skip_white"],
             f"{tag}排除近白")
    ck.check(win.skip_black_check.isChecked() == exp["skip_black"],
             f"{tag}排除近黑")
    ck.check(win.db_edit.text() == exp["db"], f"{tag}库路径")
    ck.check([win.folder_list.item(i).text()
              for i in range(win.folder_list.count())] == exp["folders"],
             f"{tag}照片文件夹 2 个")
    for name, want in exp["variants"].items():
        ck.check(win.variant_checks[name].isChecked() == want,
                 f"{tag}变体 {name}={want}")
    ck.check(win.thumb_spin.value() == exp["thumb"], f"{tag}缩略图尺寸")
    ck.check(win.dedup_check.isChecked() == exp["dedup"], f"{tag}去重开关")
    ck.check(win.subfolder_check.isChecked() == exp["subfolders"],
             f"{tag}含子文件夹")
    ck.check(win.cell_quality_spin.value() == exp["cell_quality"],
             f"{tag}JPEG 质量")
    ck.check(win.workers_spin.value() == exp["workers"], f"{tag}并行进程数")
    ck.check([win.master_queue.item(i).text()
              for i in range(win.master_queue.count())] == exp["masters"],
             f"{tag}主图队列 3 张")
    ck.check(win._master_overrides == exp["overrides"],
             f"{tag}主图单独配置恢复", repr(win._master_overrides))
    if win.master_queue.count() > 1:
        it = win.master_queue.item(1)
        ck.check(it.font().bold(), f"{tag}覆盖项队列加粗")
        ck.check("style=dot" in (it.toolTip() or ""),
                 f"{tag}覆盖项 tooltip", repr(it.toolTip()))
    ck.check(win.outdir_edit.text() == exp["out_dir"], f"{tag}默认保存到")
    ck.check(win.zoom_combo.currentText() == exp["zoom"],
             f"{tag}缩放={exp['zoom']}", win.zoom_combo.currentText())
    ck.check(win.format_combo.currentText() == exp["format"],
             f"{tag}导出格式={exp['format']}", win.format_combo.currentText())
    ck.check(win.out_quality.value() == exp["out_quality"],
             f"{tag}导出质量={exp['out_quality']}")
    ck.check(win.cell_px_spin.value() == exp["cell_px"],
             f"{tag}每格像素={exp['cell_px']}", str(win.cell_px_spin.value()))
    ck.check(win.pages.currentIndex() == exp["page"],
             f"{tag}当前页={exp['page']}", str(win.pages.currentIndex()))
    ck.check(win.master_path == exp["master"], f"{tag}当前主图恢复")
    ck.check(win.master_rot == exp["master_rot"],
             f"{tag}主图旋转角={exp['master_rot']}", str(win.master_rot))
    ck.check(win._welcome_shown == exp["welcome_shown"], f"{tag}欢迎标记")
    ck.check(win.tree is not None, f"{tag}网格树已恢复")


def main() -> int:
    ck = common.Checker("QA-B 设置记忆")
    common.get_app()
    common.patch_dialogs()

    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT)
    ctx = {
        "db": os.path.join(OUT_ROOT, "tiles.db"),
        "folder1": os.path.join(OUT_ROOT, "photos_a"),
        "folder2": os.path.join(OUT_ROOT, "photos_b"),
        "outdir": os.path.join(OUT_ROOT, "成品"),
        "masters": [],
        "settings": os.path.join(OUT_ROOT, "settings.json"),
    }
    os.makedirs(ctx["folder1"])
    os.makedirs(ctx["folder2"])
    os.makedirs(ctx["outdir"])
    for i in range(3):
        p = os.path.join(OUT_ROOT, f"master{i}.png")
        common.make_master(p, 256, 256, seed=11 + i)
        ctx["masters"].append(p)

    before = _real_settings_snapshot()

    import photomosaic.gui as gui_mod

    saved_path = gui_mod.SETTINGS_PATH
    gui_mod.SETTINGS_PATH = ctx["settings"]
    try:
        # ---------- 通道 1：closeEvent → SETTINGS_PATH → 新窗口构造读回 ----------
        ck.section("通道 1：closeEvent 写存档 → 新窗口读回")
        w1 = gui_mod.MainWindow(load_settings=True)   # 存档不存在 → 默认值起步
        exp = _set_all(w1, ctx)
        leaves_n = len(w1.tree.leaf_boxes()) if w1.tree else 0
        ck.check(leaves_n > 0, "改动后网格树非空", str(leaves_n))
        w1.close()                                    # closeEvent 写存档
        common.pump()
        ck.check(os.path.exists(ctx["settings"]),
                 "closeEvent 已写（打桩的）SETTINGS_PATH")
        with open(ctx["settings"], "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        ck.check(raw.get("style") == "dot" and raw.get("threshold") == 61.0,
                 "存档 json 含关键字段", repr({k: raw.get(k) for k in
                                               ("style", "threshold")}))
        ck.check(raw.get("masters") == ctx["masters"], "存档 json 含主图队列")

        w2 = gui_mod.MainWindow(load_settings=True)   # 构造时 _apply_state 读回
        common.pump()
        _assert_all(ck, w2, exp, "[关开]")
        w2.close()
        common.pump()

        # ---------- 通道 2：load_settings=False + 显式 _save/_apply ----------
        ck.section("通道 2：_gather_state/_apply_state 直调（不碰 SETTINGS_PATH）")
        w3 = gui_mod.MainWindow(load_settings=False)
        _set_all(w3, ctx)
        state = w3._gather_state()
        tmp_json = os.path.join(OUT_ROOT, "manual.json")
        w3._save_state(tmp_json)
        ck.check(os.path.exists(tmp_json), "显式 _save_state 落盘")
        w3.close()

        w4 = gui_mod.MainWindow(load_settings=False)
        with open(tmp_json, "r", encoding="utf-8") as fh:
            w4._apply_state(json.load(fh))
        common.pump()
        _assert_all(ck, w4, exp, "[直调]")
        w4.close()

        # ---------- 负面对照：load_settings=False 不读档 ----------
        ck.section("负面对照")
        w5 = gui_mod.MainWindow(load_settings=False)
        ck.check(w5.threshold_slider.value() == 25
                 and w5._style == "grid" and w5.pages.currentIndex() == 0,
                 "load_settings=False 保持出厂默认")
        w5.close()
    finally:
        gui_mod.SETTINGS_PATH = saved_path

    # ---------- 真实用户设置文件未被触碰 ----------
    after = _real_settings_snapshot()
    ck.check(before == after, "真实 ~/.photomosaic_studio.json 未被创建/修改",
             f"before={before} after={after}")
    ck.check(not [c for c in common.MSGBOX_CALLS if c[0] in ("warning", "critical")],
             "全程无 warning/critical 弹窗", repr(common.MSGBOX_CALLS[:3]))
    return ck.summary()


if __name__ == "__main__":
    sys.exit(common.run_guarded(main))
