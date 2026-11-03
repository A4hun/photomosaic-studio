# -*- coding: utf-8 -*-
"""QA 流程 C：③页控件改动 → 网格预览 overlay 必须联动（抓"控件改了预览没变"）。

覆盖：细节识别（梯度/显著性）、首级保守、细节阈值、风格、cells_x、
层级单选、自动细分开关。每个控件走真实 setValue/click（信号驱动），
预览端用 adaptive_leaves 独立重算做确定性对照，并验证方向性变化。
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (Checker, get_app, make_master, patch_dialogs, pump,
                    run_guarded)  # noqa: E402

OUT = os.path.join(common.ROOT, "_qa", "out", "flow_c")


def _fresh_dir(path: str) -> str:
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def expected_overlay(win):
    """按当前控件状态独立重算 overlay 应有的叶子集合。"""
    from photomosaic.render import adaptive_leaves
    if not (win.auto_check.isChecked() and win.detail_map is not None
            and win._master_pil is not None):
        return sorted((tuple(b), l) for b, l in win.tree.leaf_boxes())
    leaves = adaptive_leaves(
        win._master_pil, win.tree.leaf_boxes(), win._effective_levels(),
        float(win.threshold_slider.value()), detail=win.detail_map,
        first_level_factor=2.0 if win.first_level_check.isChecked() else 1.0)
    return sorted(((x0, y0, x1, y1), lv) for x0, y0, x1, y1, lv in leaves)


def overlay_now(win):
    return sorted((tuple(b), l) for b, l in win.grid_view._leaves)


def main() -> int:
    common.patch_dialogs()
    app = get_app()
    ck = Checker("flow_c")

    from photomosaic.gui import MainWindow
    from photomosaic.render import DetailMap
    from photomosaic.saliency import SaliencyMap

    work = _fresh_dir(OUT)
    master = make_master(os.path.join(work, "master.png"), 256, 256)

    win = MainWindow(load_settings=False)
    win.resize(1300, 900)
    win.show()
    win.prefill(master=master)
    win._goto(2)
    pump(120)
    win.cells_x_spin.setValue(8)
    win.level_radios[2].click()            # 三级
    win.auto_check.setChecked(True)
    win.threshold_slider.setValue(25)
    pump(350)                              # 阈值去抖 200ms + 余量

    ck.section("基线：auto 开，三级，阈值 25")
    ck.check(overlay_now(win) == expected_overlay(win),
             "基线 overlay == 独立重算")
    base_n = len(win.grid_view._leaves)
    ck.check(base_n > 64, "噪点圆盘驱动出细分", f"{base_n} 叶")

    ck.section("细节阈值 25 → 90（保守）→ 0（全细分）")
    win.threshold_slider.setValue(90)
    pump(350)
    n90 = len(win.grid_view._leaves)
    ck.check(overlay_now(win) == expected_overlay(win), "阈值 90：overlay 已重算")
    ck.check(n90 < base_n, "阈值 90 → 细分减少", f"{base_n} → {n90}")

    win.threshold_slider.setValue(0)
    pump(350)
    n0 = len(win.grid_view._leaves)
    ck.check(overlay_now(win) == expected_overlay(win), "阈值 0：overlay 已重算")
    ck.check(n0 > base_n, "阈值 0 → 全细分", f"{base_n} → {n0}")

    win.threshold_slider.setValue(25)
    pump(350)
    ck.check(len(win.grid_view._leaves) == base_n, "阈值回到 25 → 叶数复原")

    ck.section("首级保守")
    win.first_level_check.setChecked(True)   # toggled → 立即刷新
    pump(60)
    nfl = len(win.grid_view._leaves)
    ck.check(overlay_now(win) == expected_overlay(win),
             "首级保守：overlay 已按 ×2 门槛重算")
    ck.check(nfl <= base_n, "首级保守 → 细分不增", f"{base_n} → {nfl}")
    win.first_level_check.setChecked(False)
    pump(60)
    ck.check(len(win.grid_view._leaves) == base_n, "取消首级保守 → 复原")

    ck.section("细节识别：梯度能量 → 显著性主体 → 换回")
    ck.check(isinstance(win.detail_map, DetailMap), "初始是梯度能量 DetailMap")
    win.signal_combo.setCurrentIndex(1)      # currentIndexChanged → _on_signal_change
    pump(60)
    ck.check(isinstance(win.detail_map, SaliencyMap), "切换后是 SaliencyMap")
    ck.check(overlay_now(win) == expected_overlay(win),
             "显著性：overlay 已用新信号图重算")
    n_sal = len(win.grid_view._leaves)
    win.signal_combo.setCurrentIndex(0)
    pump(60)
    ck.check(isinstance(win.detail_map, DetailMap), "换回后是 DetailMap")
    ck.check(len(win.grid_view._leaves) == base_n, "换回梯度 → 叶数复原",
             f"显著性 {n_sal} → {len(win.grid_view._leaves)} (基线 {base_n})")

    ck.section("风格卡片：预览形状跟着变")
    win._style_buttons["hex"].click()
    pump(60)
    ck.check(win.grid_view._style_name == "hex", "hex 卡片 → 预览画六边形")
    ck.check(overlay_now(win) == expected_overlay(win),
             "切风格不改变叶子集合，只改形状")
    win._style_buttons["dot"].click()
    pump(60)
    ck.check(win.grid_view._style_name == "dot", "dot 卡片 → 预览画圆形")
    win._style_buttons["grid"].click()
    pump(60)
    ck.check(win.grid_view._style_name == "grid", "grid 卡片 → 预览画矩形")

    ck.section("层级单选：上限联动 + 手动超深叶被收回")
    win.auto_check.setChecked(False)         # 手动网格语义更直观
    pump(60)
    win.tree.split_at(32, 32)
    win.tree.split_at(32, 32)
    win._refresh_grid_overlay()
    pump(60)
    ck.check(max(l for _b, l in win.tree.leaf_boxes()) >= 3, "预置：存在 3 级深叶")
    win.level_radios[0].click()              # 一级 → set_max_level(1) 收回深叶
    pump(60)
    ck.check(max(l for _b, l in win.tree.leaf_boxes()) == 1,
             "选「一级」后深叶被收回（上限联动）")
    ck.check(overlay_now(win) == expected_overlay(win), "overlay 与树一致")
    win.level_radios[2].click()              # 回到三级
    pump(60)

    ck.section("cells_x：重建网格")
    win.cells_x_spin.setValue(16)
    pump(60)
    ck.check(len(win.tree.leaf_boxes()) == 256, "cells_x 8→16 → 256 基础块")
    ck.check(overlay_now(win) == expected_overlay(win),
             "cells_x 改动后 overlay 已重算")

    ck.section("自动细分开关")
    win.cells_x_spin.setValue(8)
    win.auto_check.setChecked(False)
    pump(60)
    n_off = len(win.grid_view._leaves)
    ck.check(n_off == 64, "auto 关 → overlay == 手动树（64 叶）", str(n_off))
    win.auto_check.setChecked(True)
    pump(60)
    n_on = len(win.grid_view._leaves)
    ck.check(overlay_now(win) == expected_overlay(win), "auto 开 → overlay 已重算")
    ck.check(n_on > n_off, "auto 开 → 幽灵格细分出现", f"{n_off} → {n_on}")

    win.close()
    return ck.summary()


if __name__ == "__main__":
    sys.exit(run_guarded(main))
