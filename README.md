# photomosaic

用大量小照片拼出一张大图（photo mosaic / 照片蒙太奇）的独立 Python 实现。
纯 Pillow + numpy + SQLite，无其他依赖。

> 本项目是对经典 photomosaic 技术的**独立干净实现**（clean-room）：算法结构参考了
> 对某商业拼图软件的技术分析（3x3 子块 LAB 特征匹配 + 近优随机选择 + MD5 去重 +
> SQLite 瓦片缓存 + 镜像/旋转变体），但代码为本项目从零编写，未复制任何原始厂商
> 代码或资源。photomosaic 技术本身属于公开的通用算法领域。

## 算法

1. **瓦片库**：扫描文件夹 → 按文件 MD5 去重 → 中心裁剪到单元格比例 → 生成变体
   （`orig / mirror / r90 / r180 / r270`）→ 每张瓦片计算特征并存入 SQLite：
   - **27 维特征**：图像盒式降采样到 3x3 子块均值 → 转 CIE LAB → 展平
     （9×L + 9×a + 9×b）。相比单一平均色，它编码了瓦片的**局部色彩结构**；
   - **平均 RGB**（3 维）：用于可选的颜色半径预筛选；
   - JPEG 缩略图（默认 256px）：渲染时直接取用，避免重复解码原图。
2. **主图网格化**：主图缩到 `cells_x × cells_y` 网格（单元格保持近似正方形），
   每格计算同样的 27 维 LAB 特征 + 平均 RGB。
3. **匹配**（每格）：
   - 距离 = 主图格与瓦片 27 维向量的**平方欧氏距离**（BLAS 矩阵向量积
     `|f|² − 2f·c + |c|²`，不开方——单调等价）；
   - 可选**半径预筛**：平均 RGB 距离超过 `--radius` 的瓦片直接排除；
   - **近优随机**：取距离最小的前 K 名（`--top-k`），从中随机抽一个——
     既忠实原图又避免同一瓦片反复出现造成的呆板；
   - **复用上限**：`--max-reuse` 限制每张瓦片使用次数（0 = 不限）。
4. **组装**：按选中瓦片贴缩略图到画布，输出 PNG/JPEG；统计平均距离、
   唯一瓦片数、变体分布。

### 色彩融合（柔光）

对齐原版观感：贴图前每个瓦片可选地向所在格的主图均色做 **SoftLight（柔光）**
混合，再按强度回混：`out = blend(瓦片, soft_light(瓦片, 格均色), 强度)`。
GUI 滑杆 0–100%、CLI `--blend 0..1`，默认 **55%**（原版的 139/255）。融合在贴图前按瓦片逐块生效。

### 原版对齐的默认值

| 项 | 默认 |
|---|---|
| 嵌套层级 | 4 级（GUI 默认选中四级） |
| 基础块总数 | ≈300（自动按主图比例换算横向块数） |
| 输出/库 JPEG 质量 | 100 |
| 近白/近黑瓦片排除 | 可勾选（亮度阈值 245/10） |
| 首级保守 | 可勾选（首级细分门槛 ×2，大块不轻易切碎） |

### 自适应多级（细节阶梯）

`--levels N`（N>1）启用四叉树细分：以基础网格为第 1 级，对**局部细节能量**
（32x32 亮度梯度均值）超过 `--detail-threshold` 的格子递归切成 2x2，直到第 N 级
或格子小于 16px。效果：眼睛/纹理等细节区由更多更小的瓦片组成，平坦区保持大
瓦片——即经典商业拼图软件的"多级阶梯"机制。自适应模式画布尺寸 = 主图尺寸。

### 手动区域强制细化

在自动阶梯之外，可圈定矩形**强制**细分到指定级别（无视能量阈值，对应原软件
"手动设置某区域细节"的能力）：

```powershell
# CLI：可重复 --region，坐标为主图像素，@级别 可选（默认 2）
.venv\Scripts\python -m photomosaic make 主图.jpg --db tiles.db -o out.png ^
    --levels 3 --region 512,128,768,384@4 --region 100,100,200,200@3

# 或 JSON 文件：[{"box":[x0,y0,x1,y1],"level":4}, ...]
.venv\Scripts\python -m photomosaic make 主图.jpg --db tiles.db -o out.png --regions-file regions.json
```

GUI 步骤③的网格编辑器（叠加层粗线为手动块、细线为自动细分）：

| 操作 | 效果 |
|---|---|
| 左键 | 分割一格（+1 细节） |
| Ctrl+左键 | 该块**细分到底**（最大层级） |
| Ctrl+左键拖框 | 橡皮筋圈范围，框内**全部细分到所选嵌套层级** |
| 右键 | **取消该区域细节**（整块折叠回基础格） |
| Ctrl+右键 | 只合并一级（细粒度回退） |

CLI 也可用可重复参数 `--region X0,Y0,X1,Y1[@LEVEL]` 批量指定，即使
`--levels 1`，只要存在区域也会启用自适应画布。

## 断点续传与记忆

- **扫描续传**：`file_index` 表缓存 (路径, mtime, size → MD5)，未改动文件免重哈希；
  已入库 MD5 直接跳过。GUI 支持**暂停/继续/取消**；取消或崩溃后再次构建即从
  断点继续（按批 512 行提交，丢失上限为最后一批）。
- **生成续传**：自适应/手动模式渲染时写 checkpoint 目录（默认 `<输出>.ckpt`）：
  参数指纹 + 每格选片 + 画布快照。同参数重跑自动续传（状态栏显示"续传跳过 N 块"），
  "取消生成"保留进度，成功后 checkpoint 自动删除。
- **会话记忆**：关闭时自动保存全部设置/主图/库/文件夹/网格树/窗口几何
  （位置尺寸与最大化状态）/第③页分栏比例到 `~/.photomosaic_studio.json`，
  下次启动恢复（窗口完全移出屏幕时回默认 1280×820）。
- **工程文件**：文件菜单 保存项目/打开项目（`.pmproj`，JSON），含网格树与全部参数，
  可跨机器迁移（路径需自行一致或重选）。

### 超大分辨率输出（分块流式渲染）

输出超过 **2 亿像素** 时自动切换为分条带流式渲染：画布按水平条带（内存预算
约 384MB/条）逐条分配-贴图-落盘-释放，峰值内存与输出总面积**解耦**——几百
亿像素也能稳定跑完。特性：

- 输出为 **BigTIFF**（自研纯 Python strip 写入器，RGB 8bit 无压缩，专业看图
  软件可直接打开；输出名自动改为 `.tif`）；文件布局预分配，**崩溃/取消后
  重跑同参数原位续传**（条带级断点）；
- 与整图路径**逐像素一致**（瓦片全局解码尺寸与 JPEG draft 缩放级别完全对齐，
  测试逐像素断言）；
- 同时生成 `*.preview.png`（≤4096px 宽降采样预览），GUI 大图预览走它，不把
  整图读进内存；
- **应用内分段查看**（GUI 第④页"🔍 放大查看细节"）：几十 GB 的成品也能在
  应用内平移/缩放浏览——缩小画预览，放大超过预览分辨率时按需从 BigTIFF 读
  条带拼出视口（1:1 像素，滚轮缩放/拖拽平移/双击适应或 100%）；
- CLI 可用 `--out-width` 指定宽度触发；GUI"自定义宽度"上限 1,000,000px，
  超阈值时实时提示"将用分块流式渲染，输出 BigTIFF"。

实测（4080 Laptop / 40 张 512px 瓦片库 / 93,397 格四级细分）：3.41 亿像素
（16000×21333）→ 114 秒、峰值内存 2.3GB、输出 1GB；同参数下 300 亿像素约
300 条带，峰值内存不变，仅时间与磁盘线性增长。

## 图形界面

```powershell
.venv\Scripts\pip install pyside6
.venv\Scripts\python -m photomosaic gui
```

主窗口：左侧主图预览，右侧 Settings / Result 页签（瓦片库构建、网格与自适应
级别、匹配参数），底部进度条与状态栏；ingest 与 render 在后台线程执行。

## 安装

```powershell
cd D:\PyCharmProject\project\photomosaic
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt pytest
```

## 使用

```powershell
# 一步式：临时库 + 渲染
.venv\Scripts\python -m photomosaic run 主图.jpg D:\我的照片 -o mosaic.png --cells-x 120 --variants orig,mirror

# 两步式：先建瓦片库（可复用，增量去重）
.venv\Scripts\python -m photomosaic ingest D:\我的照片 --db tiles.db --variants orig,mirror,r90,r180,r270
# 再渲染（换主图无需重建库）
.venv\Scripts\python -m photomosaic make 主图.jpg --db tiles.db -o mosaic.png --cells-x 200 --top-k 6 --max-reuse 50 --seed 42
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--cells-x` | 网格宽度（格数）；越大越细腻，瓦片库要求越高 |
| `--cell-px` / `--tile-px` | 输出每格像素尺寸 |
| `--top-k` | 近优随机候选数（1 = 总取最优，越大越随机） |
| `--max-reuse` | 单瓦片最大复用次数（0 不限） |
| `--radius` | 平均色预筛半径（0-441，RGB 欧氏距离） |
| `--seed` | 随机种子（同种子同库 ⇒ 输出逐字节确定） |
| `--blend` | 柔光融合强度 0–1（默认 0.55 ≈ 原版 139/255；0 关闭） |
| `--out-width` | 输出宽度（像素），高按主图比例；适合海报放大 |
| `--stream-threshold` | 输出像素超过该值切换为流式 BigTIFF 渲染（默认 2 亿） |
| `--stream-strip-rows` | 调试/测试用：强制流式条带高度（默认按内存预算自动） |
| `--poster` | `2x3` / `3x2`：成品另存分页到 `<输出名>_poster/` |
| `--variants` | 瓦片变体：`orig,mirror,r90,r180,r270` |
| `--ratio` | 单元格宽高比（默认 1.0 正方形） |

演示（生成合成数据并出图）：

```powershell
.venv\Scripts\python scripts\demo.py
```

## 测试

```powershell
.venv\Scripts\python -m pytest tests -q
```

## 已知限制

- 流式超大输出（BigTIFF）暂不支持 GUI 的点击替换碎片与非 TIFF 格式导出
  （状态栏有提示；如需替换/导出其他格式，请降低输出尺寸走整图路径）；
- 亿级**格子数**下的统计内存未优化：流式渲染解决的是像素数维度
  （画布内存），不是格子数维度（逐格统计结构仍全量驻留）；
- 套索"取消细节"的语义为**相交即合并**：与套索多边形相交的块都会被
  合并回父块，不做精确轮廓扣除。

## 性能与规模

- **GPU 加速**：匹配的距离矩阵 GEMM + top-k 在装有 CuPy 时自动跑在 GPU
  （`--device auto|cpu|gpu`，GUI 里"计算设备"下拉）。RTX 4080 Laptop 实测
  10 万格 × 5 万瓦片：**GPU 6.4s（1.57 万格/s）vs CPU 76s**；`auto` 有卡即用卡。
- 特征提取走**积分图**（半/四分之一分辨率 LAB+RGB 积分），任意矩形 3x3 子块均值
  O(1)，全格向量化，无逐格 Pillow 裁剪。
- 距离矩阵按 chunk 计算（CPU 512 行 / GPU 4096 行一块），峰值内存 ≈ chunk×N×4B，
  不随格子总数增长。
- 缩略图 LRU 缓存默认 512 张（`cache_cap` 可调）；GUI 结果图不常驻内存
  （预览走磁盘 QPixmap，替换/导出按需加载即用即释）。
- 瓦片特征矩阵内存 ≈ `N × (27+3) × 4B`（百万瓦片约 120MB）。
- 内存排查提示：若整机内存居高不下，先检查无关的构建守护进程
 （Kotlin/Gradle daemon 空闲常驻数 GB 很常见，`taskkill /F /IM java.exe` 或
  `gradle --stop` 可回收，下次构建自动重启）。

## 项目结构

```
photomosaic/
  features.py   # 27 维 3x3-LAB 特征、平均色、比例裁剪
  db.py         # SQLite 瓦片缓存（特征+缩略图+MD5）
  library.py    # 文件夹扫描、去重、变体、入库
  match.py      # 半径预筛 + 平方距离 + top-k 近优随机 + 复用上限
  planner.py    # 智能分配（复用均衡 + 邻域去重）
  grid.py       # 四叉树网格编辑模型（含自由套索选区）
  render.py     # 主图网格化、组装、统计（含流式渲染接入）
  bigtiff.py    # 纯 Python BigTIFF strip 写入器（可续写）+ 分段读取器
  streaming.py  # 分条带流式渲染（百亿像素级）
  bigview.py    # 超大成品分段查看对话框（预览打底 + 按需读条带）
  cli.py        # ingest / make / run 子命令
tests/          # 特征、匹配、端到端、确定性、套索、流式、查看器、布局测试
scripts/demo.py # 合成数据演示
```
