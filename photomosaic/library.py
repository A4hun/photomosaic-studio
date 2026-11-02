"""Tile library ingestion: scan folders, deduplicate by MD5, generate
orientation/mirror variants, extract features and cache thumbnails.

The per-file pipeline (decode -> crop -> variants -> resize -> feature ->
JPEG encode) is CPU-bound and runs in a process pool so all cores are used;
the SQLite writer stays in the parent process and commits in batches.
JPEG sources are decoded at reduced scale via PIL draft mode, which is
visually lossless for thumbnail-sized outputs but several times faster.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from .db import TileDB
from .features import average_rgb, crop_to_ratio, encode_vector, subblock_lab


class IngestControl:
    """Pause / cancel controls shared with the ingest worker loop."""

    def __init__(self) -> None:
        self.paused = threading.Event()
        self.cancelled = threading.Event()

    def pause(self) -> None:
        self.paused.set()

    def resume(self) -> None:
        self.paused.clear()

    def cancel(self) -> None:
        self.cancelled.set()
        self.paused.clear()

    def wait_while_paused(self, timeout: float = 0.2) -> None:
        while self.paused.is_set() and not self.cancelled.is_set():
            time.sleep(timeout)

VARIANTS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "orig": lambda im: im,
    "mirror": lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    "r90": lambda im: im.transpose(Image.Transpose.ROTATE_90),
    "r180": lambda im: im.transpose(Image.Transpose.ROTATE_180),
    "r270": lambda im: im.transpose(Image.Transpose.ROTATE_270),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def iter_images(folder: str, recursive: bool = True) -> Iterable[str]:
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    yield os.path.join(root, name)
    else:
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield path


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _thumb_for(img: Image.Image, variant: str, ratio: float, thumb_size: int) -> Image.Image:
    """Apply the variant transform, then re-crop to the cell ratio so every
    stored thumbnail has exactly the cell aspect ratio (no stretch later)."""
    base = VARIANTS[variant](img)
    base = crop_to_ratio(base, ratio)
    w = max(1, thumb_size)
    h = max(1, int(round(thumb_size / ratio)))
    return base.resize((w, h), Image.Resampling.LANCZOS)


def _process_file(args):
    """Worker: full per-file pipeline. Module-level for pickling (spawn).

    args = (path, ratio, variants, thumb_size, jpeg_quality,
            skip_near_white, skip_near_black, mtime, size).
    The file is read from disk exactly once: MD5 and decoding share the bytes.
    Returns (status, path, rows, meta) where meta = (mtime, size, md5) for the
    resumable file index (None if the file could not even be read).  When a
    threshold is set, files whose average luminance exceeds it (white) or falls
    below it (black) are skipped entirely (all variants).
    """
    (path, ratio, variants, thumb_size, jpeg_quality,
     skip_near_white, skip_near_black, mtime, size) = args
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return "unreadable", path, [], None
    md5 = hashlib.md5(data).hexdigest()
    meta = (mtime, size, md5)
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format == "JPEG":
                # reduced-scale decode: several times faster, imperceptible at thumb size
                im.draft("RGB", (thumb_size * 2, thumb_size * 2))
            im.load()
            rgb = im.convert("RGB")
    except Exception:
        return "unreadable", path, [], meta
    del data
    cropped = crop_to_ratio(rgb, ratio)
    if skip_near_white is not None or skip_near_black is not None:
        avg = average_rgb(cropped)
        lum = 0.2126 * float(avg[0]) + 0.7152 * float(avg[1]) + 0.0722 * float(avg[2])
        if skip_near_white is not None and lum > skip_near_white:
            return "skipped_white", path, [], meta
        if skip_near_black is not None and lum < skip_near_black:
            return "skipped_black", path, [], meta
    rows = []
    for variant in variants:
        thumb = _thumb_for(cropped, variant, ratio, thumb_size)
        feat = subblock_lab(thumb)
        avg = average_rgb(thumb)
        buf = io.BytesIO()
        thumb.save(buf, "JPEG", quality=jpeg_quality)
        rows.append((md5, variant, path, thumb.width, thumb.height,
                     encode_vector(feat), encode_vector(avg), buf.getvalue()))
    return "ok", path, rows, meta


def default_workers() -> int:
    """All logical cores (Windows caps a process pool at 61 workers)."""
    return max(1, min(os.cpu_count() or 4, 61))


def ingest(
    db: TileDB,
    folder: str,
    ratio: float = 1.0,
    recursive: bool = True,
    variants: Sequence[str] = ("orig",),
    thumb_size: int = 256,
    dedup: bool = True,
    jpeg_quality: int = 100,
    workers: int = 0,
    skip_near_white: Optional[int] = None,
    skip_near_black: Optional[int] = None,
    control: Optional[IngestControl] = None,
    progress: Callable[[int, str], None] | None = None,
    on_phase: Callable[[str, int, int], None] | None = None,
) -> Dict[str, int]:
    """Add every image under `folder` to the tile database (parallel, resumable).

    Resumability: a file_index table maps (path, mtime, size) -> md5 so unchanged
    files are never re-read; tiles already in the database are skipped. An
    interrupted or cancelled run therefore continues where it left off when
    called again.

    Scanning only lists + stats files (cheap); hashing happens inside the worker
    processes together with decoding, so every core is busy from the first file
    and each file is read once. One persistent process pool is fed through a
    bounded in-flight window (no per-batch restarts, no end-of-wave stalls).

    `on_phase(phase, done, total)` reports coarse progress for UIs: phase
    `scan` (files seen so far, total 0 = unknown) then `build`
    (files processed, total = number of files to process).
    """
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; choose from {sorted(VARIANTS)}")
    ctrl = control or IngestControl()

    known = db.existing_md5s() if dedup else set()
    index = db.get_file_index()
    new_index: List[tuple] = []
    tasks: List[tuple] = []
    stats = {"files": 0, "added": 0, "duplicates": 0, "unreadable": 0, "rows": 0,
             "index_hits": 0, "cancelled": 0, "skipped_white": 0, "skipped_black": 0}
    for path in iter_images(folder, recursive):
        stats["files"] += 1
        if on_phase and stats["files"] % 200 == 0:
            on_phase("scan", stats["files"], 0)
        if ctrl.cancelled.is_set():
            stats["cancelled"] += 1
            continue
        try:
            st = os.stat(path)
        except OSError:
            stats["unreadable"] += 1
            continue
        hit = index.get(path)
        if hit and abs(hit[0] - st.st_mtime) < 1e-6 and hit[1] == st.st_size:
            stats["index_hits"] += 1
            if dedup and hit[2] in known:
                stats["duplicates"] += 1
                continue
        tasks.append((path, ratio, tuple(variants), thumb_size, jpeg_quality,
                      skip_near_white, skip_near_black, st.st_mtime, st.st_size))

    total = len(tasks)
    if on_phase:
        on_phase("scan", stats["files"], 0)
        on_phase("build", 0, total)
    n_workers = workers if workers and workers > 0 else default_workers()
    n_workers = max(1, min(n_workers, total or 1))
    batch: List[tuple] = []

    def flush() -> None:
        if batch or new_index:
            if batch:
                db.add_tiles(batch)
                batch.clear()
            db.update_file_index(new_index[:])
            new_index.clear()
            db.commit()

    done = 0

    def consume(item) -> None:
        nonlocal done
        done += 1
        meta = item[3]
        if meta is not None:
            new_index.append((item[1], meta[0], meta[1], meta[2]))
        _consume(item[:3], stats, batch, known)
        if len(batch) >= 512:
            flush()
        if progress:
            progress(done, item[1])
        if on_phase:
            on_phase("build", done, total)

    if n_workers == 1 or total <= 4:
        for i, t in enumerate(tasks):
            ctrl.wait_while_paused()
            if ctrl.cancelled.is_set():
                stats["cancelled"] += total - i
                break
            consume(_process_file(t))
    elif total:
        from concurrent.futures import FIRST_COMPLETED, wait

        window = n_workers * 3          # enough queued work to keep every core fed
        it = iter(tasks)
        pending = set()
        submitted = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            exhausted = False
            while True:
                if ctrl.cancelled.is_set():
                    break
                ctrl.wait_while_paused()
                while not exhausted and len(pending) < window and not ctrl.paused.is_set():
                    t = next(it, None)
                    if t is None:
                        exhausted = True
                        break
                    pending.add(ex.submit(_process_file, t))
                    submitted += 1
                if not pending:
                    break
                finished, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in finished:
                    consume(fut.result())
            if ctrl.cancelled.is_set():
                # drop queued work, keep whatever is already running/finished
                for fut in list(pending):
                    if fut.cancel():
                        pending.discard(fut)
                for fut in pending:
                    consume(fut.result())
        stats["cancelled"] += total - done
    flush()
    return stats

def _consume(item, stats: Dict[str, int], batch: List[tuple], known: set) -> None:
    status, path, rows = item
    if status == "unreadable":
        stats["unreadable"] += 1
        return
    if status in ("skipped_white", "skipped_black"):
        stats[status] += 1
        return
    md5 = rows[0][0] if rows else None
    if md5 in known:          # same content submitted twice in this run
        stats["duplicates"] += 1
        return
    if md5:
        known.add(md5)
    batch.extend(rows)
    stats["rows"] += len(rows)
    stats["added"] += 1
