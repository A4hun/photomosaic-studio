# _qa/ — 离屏 GUI 真实流程 QA 脚本

每个脚本覆盖一条真实用户流程，含断言，可独立运行。一律
`QT_QPA_PLATFORM=offscreen` 离屏（绝不弹真实窗口）；弹窗统一打桩记录。
合成素材由 numpy+PIL 现场生成，产物写在 `_qa/out/<flow>/`（可删除）。

## 运行

```powershell
# 仓库根目录；任一脚本退出码 0=全过，1=有断言失败，2=脚本异常
.venv\Scripts\python _qa\qa_flow_a_pipeline.py
.venv\Scripts\python _qa\qa_flow_b_gestures.py
.venv\Scripts\python _qa\qa_flow_c_consistency.py
```

## 覆盖

- **qa_flow_a_pipeline.py**（38 断言）：①打开主图 → ②点真实"构建"按钮建库
  → ③调 cells_x/层级/风格 → ④点真实"生成"按钮。断言产出存在、尺寸=
  cells_x×每格像素、一级均匀网格叶数、不同 cells_x 格数应变；
  hex 风格格子角落精确等于高斯模糊背景底（六边形缝隙）、grid 角落是实心
  瓦片、brick 偶数行右移半格。
- **qa_flow_b_gestures.py**（25 断言）：③页 GridView 用 QTest 真实鼠标事件
  跑全部手势（左键细分/右键合并/Ctrl+左键套索/Ctrl+右键套索/
  Ctrl+Shift 矩形框选双向/Ctrl+左键到最细/Ctrl+右键取消整块），每个手势
  后断言树的叶子数与层级；随后点④生成，断言产出叶子集合与编辑后的树
  逐一相等（手动网格不被自动覆盖）；自动细分开着时首次编辑物化
  （before_edit → _materialize_auto_grid）后右键取消必须生效；物化后
  粒度上限不得越过所选级数。
- **qa_flow_c_consistency.py**（27 断言）：细节识别（梯度/显著性）、首级
  保守、细节阈值、风格、cells_x、层级单选、自动开关——每次真实控件改动
  后，overlay 叶子集合必须等于按当前控件状态独立重算的结果
  （抓"控件改了预览没变"），并验证阈值/首级保守的方向性效果。

公共助手见 `common.py`（离屏引导、弹窗打桩、素材合成、worker 等待、
断言记账）。相关单元级回归在 `tests/test_qa_regressions.py`。
