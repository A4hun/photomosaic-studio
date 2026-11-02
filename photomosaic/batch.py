"""批量出图：库只加载一次（Matcher 含 GPU 特征常驻），逐主图×风格渲染。

核心收益：大瓦片库（几十万张）加载一次即可，不逐张重载。支持：
- 主图矩阵：多张主图 × 多种风格 × 可选参数覆盖
- 智能适配：每张主图先 analyze → 推荐参数（可用 auto=True 关闭用统一参数）
- 可靠性：单张失败不拖垮整批（记日志继续）；已存在输出自动跳过（断点续批）
- 进度：双层（第 N/M 张 + 当前张阶段）
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

from PIL import Image

from .db import TileDB
from .match import Matcher
from .render import make_mosaic
from .tuning import recommend_params

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def scan_masters(folder: str, recursive: bool = True) -> List[str]:
    """扫描文件夹里的主图。"""
    out: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    out.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(folder, f))
    return sorted(out)


class BatchRunner:
    """库常驻的批量渲染器。"""

    def __init__(self, db_path: str, device: str = "auto"):
        self.db = TileDB(db_path)
        self.tiles = self.db.load_arrays()           # 库只加载一次
        self.matcher = Matcher(self.tiles, device=device)  # GPU 特征常驻
        self.n_tiles = int(self.matcher.n)

    def close(self) -> None:
        try:
            self.matcher.release()
        except Exception:
            pass
        self.db.close()

    def __enter__(self) -> "BatchRunner":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def run(
        self,
        masters: List[str],
        out_dir: str,
        styles: Optional[List[str]] = None,
        auto_adapt: bool = True,
        overwrite: bool = False,
        base_params: Optional[Dict] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
        status: Optional[Callable[[str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, object]:
        """批量渲染 masters × styles。返回统计。

        auto_adapt=True 时每张主图先 analyze → 推荐参数（智能适配）；
        False 时用统一 base_params。
        """
        os.makedirs(out_dir, exist_ok=True)
        styles = styles or ["grid"]
        jobs = [(m, st) for m in masters for st in styles]
        total = len(jobs)
        done = ok = skipped = failed = 0
        errors: List[str] = []
        per_image: List[Dict] = []
        t_start = time.time()

        for master_path, style in jobs:
            if should_cancel and should_cancel():
                break
            stem = os.path.splitext(os.path.basename(master_path))[0]
            suffix = f"_{style}" if style != "grid" else ""
            out_path = os.path.join(out_dir, f"{stem}{suffix}_mosaic.png")
            if not overwrite and os.path.exists(out_path):
                skipped += 1
                done += 1
                if progress:
                    progress(done, total, master_path)
                continue
            if status:
                status(f"[{done + 1}/{total}] {os.path.basename(master_path)}"
                       f" · {style}")
            try:
                with Image.open(master_path) as im:
                    im.load()
                    master_img = im.convert("RGB")
                params = dict(base_params or {})
                if auto_adapt:
                    rec = recommend_params(master_img, style)
                    reasons = rec.pop("reasons")
                    rec.pop("_features", None)
                    rec.pop("style", None)
                    # skip_near_white/black 是建库（ingest）参数，不是渲染参数——剔除
                    rec.pop("skip_near_white", None)
                    rec.pop("skip_near_black", None)
                    params.update(rec)
                # 风格必须在参数里（auto_adapt 时 recommend 的 style 已被剔除，这里补回）
                params["style"] = style
                t0 = time.time()
                stats = make_mosaic(master_path, self.db, out_path,
                                    shared_matcher=self.matcher,
                                    should_cancel=should_cancel, **params)
                dt = time.time() - t0
                ok += 1
                per_image.append({
                    "master": master_path, "style": style, "output": out_path,
                    "seconds": round(dt, 1), "size": stats.get("size"),
                })
            except Exception as exc:  # noqa: BLE001  单张失败不拖垮整批
                failed += 1
                errors.append(f"{os.path.basename(master_path)} [{style}]: "
                              f"{type(exc).__name__}: {exc}")
            done += 1
            if progress:
                progress(done, total, master_path)

        return {
            "total": total, "done": done, "ok": ok,
            "skipped": skipped, "failed": failed,
            "errors": errors, "per_image": per_image,
            "seconds": round(time.time() - t_start, 1),
            "out_dir": os.path.abspath(out_dir),
        }
