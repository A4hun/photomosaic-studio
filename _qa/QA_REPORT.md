# QA 汇报：真实用户流程离屏验证（批量出图 / 设置记忆 / 超大流式输出）

日期：2026-11-03 · 环境：Windows + Python 3.13.3 + PySide6 6.12.0（QT_QPA_PLATFORM=offscreen，全程无真实窗口）

## 交付脚本（_qa/，各自可独立运行）

| 脚本 | 覆盖流程 | 结果 |
|---|---|---|
| `_qa/qa_batch_render.py` | A. 批量出图全流程（GUI 真驱动） | 45/45 通过 |
| `_qa/qa_settings_memory.py` | B. 设置记忆全流程（关窗→重开） | 92/92 通过 |
| `_qa/qa_streaming_output.py` | C. 超大流式输出（BigTIFF/PNG/续传） | 16/16 通过 |
| `_qa/_probe.py` | 确定性/流式一致性探针（证据留存） | 通过 |

运行方式：`.venv\Scripts\python _qa\qa_xxx.py`（脚本内置 `if __name__ == "__main__"` 守卫；建库的多进程池依赖它）。

注意：`_qa/common.py` 与同目录 `qa_flow_a_pipeline.py`/`qa_flow_b_gestures.py` 是另一位 QA agent 的产物，我的三个脚本复用了它的 common.py（离屏/弹窗打桩/worker 等待），未改动它。

## Bug 清单

### Bug#1（已修复，含回归测试）：批量"单独配置"改风格后，输出文件名与内容不符

- **复现**：① 队列加 ≥2 张主图 → ③ 统一风格选「六边形蜂窝」→ 对其中一张用「单独配置」改成「圆点点彩」→ ④ 生成。
- **期望**：那张的产出文件名带 `_dot` 后缀（与实际渲染风格一致）。
- **实际（修复前）**：产出 `m1_hex_mosaic.png` —— 内容是点彩、名字写蜂窝（QA-A 首跑 FAIL 项复现，像素级对比证实内容确实是 dot）。
- **根因**：`batch.py` 的 `BatchRunner.run` 在合并 overrides **之前**就用任务风格算好了 `out_path`，文件名永远跟任务矩阵风格走。
- **修复**：命名改为跟随"实际生效风格"（`get_style(override.style or job_style).name`；只读字典、不开图，跳过判断零成本）；`per_image` 日志的 style 字段同步改成生效风格。同根因残留一起修。
- **回归测试**：`tests/test_batch.py::test_batch_override_style_reflected_in_filename`（含续跑跳过一致性）。
- **影响面**：CLI batch 不传 overrides，行为不变；GUI 批量立刻受益。

### Bug#2（已上报，未修——需 Lead 决策）：hex/dot 风格流式渲染与整图路径不逐像素一致

- **复现**：`make_mosaic(..., style="hex"|"dot", stream_threshold_px=1)` 与同参同种子整图路径对比（`_qa/_probe.py` / QA-C 第 4 节量化）。
- **实际**：缝隙（瓦片蒙版外露出背景底的区域）两处不同，本素材下 max_diff=124（probe 素材下 200）。条带边界附近尤甚。
- **根因**：有缝隙风格的背景 = 主图放大后高斯模糊。整图路径对 (OW,OH) 整图做一次模糊；流式路径在 `streaming.py` 中对每条带**单独**裁主图行区、resize、模糊——① 模糊半径取 `min(OW, strip_h)//64` 而非 `min(OW, OH)//64`（条带越矮模糊越弱）；② 条带裁剪丢掉了上下文的行贡献，条带接缝处必然有缝。
- **为什么没顺手修**：要做到逐像素一致需要"整图背景模糊一次再切条带"（OW×OH 画布常驻，违背流式初衷）或"带上下文的条带模糊 + 与整图一致的半径"（BOX resize 的行相位问题，非平凡且有风险）。建议路线：a) 半径按 (OW,OH) 统一 + 条带上下各扩 4σ 上下文再裁回（近似消除，需像素级验证）；或 b) 文档声明为已知限制。**由 Lead 拍板。**
- **影响面**：仅 style ∈ {hex, dot} 且输出超 2 亿像素走流式时；grid/brick 无背景模糊，不受影响（QA-C 已证 grid 流式==整图）。

## 各流程验证要点

### A. 批量出图（`_qa/qa_batch_render.py`，45 项断言）
- ① 队列加 3 张主图（模拟 `_open_master` 的入队逻辑）→ ② 点按钮建库（真走 `_Worker` 线程，12 图 × 2 变体 = 24 行）→ ③ 设 hex/cells_x=8/显著性/二级/种子 42 → 其中一张用「单独配置」对话框（`QDialog.exec` 打桩成选 dot+块数 10+每格像素 40 后接受）→ ④ 点「生成」。
- 间谍 BatchRunner 证实：队列 >1 自动走批量；`base_params` 与③页逐项一致；`cell_px` 被 `_batch_overrides` 正确换算成 `out_width=400`。
- 产出：3 张、命名后缀 = 实际风格、尺寸正确（两张 240×180、覆盖那张 400×300）；每张与同参直接 `make_mosaic` 的参照渲染**逐像素一致**；hex 格角像素验证为低饱和模糊底色（缝隙背景真的在）。
- 中断/续批：第 1 张完成后置取消（worker 线程内同步翻转，无竞态）→ 只产出 1 张；重跑 skipped=1、补齐 3 张。

### B. 设置记忆（`_qa/qa_settings_memory.py`，92 项断言）
- 双通道验证：通道 1 真走 `closeEvent → SETTINGS_PATH → 新窗口构造读回`（`photomosaic.gui.SETTINGS_PATH` 打桩到临时文件）；通道 2 `load_settings=False` + 显式 `_save_state/_apply_state` 到临时 json。
- 30+ 项设置逐项断言：阈值/风格(含卡片选中态)/融合/细节识别/块数/层级/自动细分/top_k/复用/半径/种子/设备/首级保守/近白近黑/库路径/文件夹/5 个变体/缩略图/去重/子文件夹/JPEG 质量/进程数/主图队列/单独配置(含加粗+tooltip)/默认保存到/缩放/导出格式/导出质量/每格像素/当前页/当前主图/旋转角/欢迎标记/网格树。
- 安全：断言真实 `~/.photomosaic_studio.json` 未被创建或修改（前后 mtime+size 快照一致）。
- 负面对照：`load_settings=False` 保持出厂默认。

### C. 超大流式输出（`_qa/qa_streaming_output.py`，16 项断言）
- `stream_threshold_px=1` 强制流式：BigTIFF 与 PNG 各出一张，tifffile/PIL 读回后与整图路径**逐像素一致**（uniform 与 adaptive+显著性 两条路径都验证）。
- BigTIFF 断点续传：条带阶段取消（status 武装 + progress 触发，只在中途取消）→ `meta.json` 记 `done_strips=3`、部分文件已写出 → 间谍证实重跑传入 `done_strips=3`（不是从头重渲）→ 成品与一次跑完逐像素一致 → `.ckpt` 成功后被清理。
- PNG：取消后重跑从头写（zlib 不可定位），成品仍与一次跑完一致。

## 测试套件
- 全量：`pytest tests -q` → **179 passed, 1 skipped**（基线 173 + 我的 1 个回归 + 其他 QA agent 的回归）。
- 兄弟 agent 的 `_qa/qa_flow_a_pipeline.py`(38/38)、`_qa/qa_flow_b_gestures.py`(25/25) 在我的改动后重跑仍全绿。

## 备注 / 环境观察（非 bug）
- 离屏 Qt 打 `QFontDatabase: Cannot find font directory ...` 到 stderr：PySide6 6.12 不再附带字体，纯提示，不影响渲染与断言。
- GUI 风格卡片在主图+库就绪后于 daemon 线程渲染小样例，与本流程并发安全（SQLite WAL 多读）。
