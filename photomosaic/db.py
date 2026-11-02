"""SQLite tile cache.

Stores one row per (source image, variant): the 27-dim LAB feature, the
average RGB, a JPEG thumbnail cropped to the cell aspect ratio, and the
source MD5 used for deduplication.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional

import numpy as np

from .features import decode_vector

SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    md5     TEXT    NOT NULL,
    variant TEXT    NOT NULL,
    path    TEXT    NOT NULL,
    width   INTEGER NOT NULL,
    height  INTEGER NOT NULL,
    feat    BLOB    NOT NULL,
    avg     BLOB    NOT NULL,
    thumb   BLOB    NOT NULL,
    UNIQUE(md5, variant)
);
CREATE INDEX IF NOT EXISTS idx_tiles_md5 ON tiles(md5);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS file_index (
    path  TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size  INTEGER NOT NULL,
    md5   TEXT NOT NULL
);
"""


class TileDB:
    def __init__(self, path: Optional[str] = None):
        self.path = path or ":memory:"
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")

    # ---- writes --------------------------------------------------------
    def has_md5(self, md5: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM tiles WHERE md5 = ? LIMIT 1", (md5,))
        return cur.fetchone() is not None

    def existing_md5s(self) -> set:
        return {r[0] for r in self.conn.execute("SELECT md5 FROM tiles")}

    def add_tile(self, md5: str, variant: str, path: str, width: int, height: int,
                 feat: bytes, avg: bytes, thumb: bytes) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO tiles(md5, variant, path, width, height, feat, avg, thumb)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (md5, variant, path, width, height, feat, avg, thumb),
        )

    def add_tiles(self, rows: Sequence[tuple]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO tiles(md5, variant, path, width, height, feat, avg, thumb)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    # ---- resume index ----------------------------------------------------
    def get_file_index(self) -> Dict[str, tuple]:
        return {
            r[0]: (r[1], r[2], r[3])
            for r in self.conn.execute("SELECT path, mtime, size, md5 FROM file_index")
        }

    def update_file_index(self, rows: Sequence[tuple]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO file_index(path, mtime, size, md5) VALUES (?, ?, ?, ?)",
            rows,
        )

    def commit(self) -> None:
        self.conn.commit()

    # ---- reads ---------------------------------------------------------
    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]

    def load_arrays(self) -> Dict[str, object]:
        """Load ids/features/averages/metadata for matching (thumbs stay in DB)."""
        rows = self.conn.execute(
            "SELECT id, md5, variant, path, feat, avg FROM tiles ORDER BY id"
        ).fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        feats = (
            np.stack([decode_vector(r[4]) for r in rows]).astype(np.float32)
            if rows
            else np.zeros((0, 27), dtype=np.float32)
        )
        avgs = (
            np.stack([decode_vector(r[5]) for r in rows]).astype(np.float32)
            if rows
            else np.zeros((0, 3), dtype=np.float32)
        )
        metas: List[dict] = [
            {"id": r[0], "md5": r[1], "variant": r[2], "path": r[3]} for r in rows
        ]
        return {"ids": ids, "feats": feats, "avgs": avgs, "metas": metas}

    def get_thumb(self, tile_id: int) -> bytes:
        row = self.conn.execute(
            "SELECT thumb FROM tiles WHERE id = ?", (tile_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"tile id {tile_id} not found")
        return row[0]

    def get_sources(self, tile_ids) -> Dict[int, tuple]:
        """tile id -> (source path, variant, thumb width, thumb height)."""
        ids = [int(i) for i in tile_ids]
        out: Dict[int, tuple] = {}
        for s in range(0, len(ids), 500):
            part = ids[s:s + 500]
            marks = ",".join("?" * len(part))
            for tid, path, variant, w, h in self.conn.execute(
                    f"SELECT id, path, variant, width, height FROM tiles WHERE id IN ({marks})",
                    part):
                out[tid] = (path, variant, w, h)
        return out

    def get_thumbs(self, tile_ids) -> Dict[int, bytes]:
        """Fetch many thumbnails with a few IN queries (instead of one per tile)."""
        ids = [int(i) for i in tile_ids]
        out: Dict[int, bytes] = {}
        for s in range(0, len(ids), 500):
            part = ids[s:s + 500]
            marks = ",".join("?" * len(part))
            for tid, raw in self.conn.execute(
                    f"SELECT id, thumb FROM tiles WHERE id IN ({marks})", part):
                out[tid] = raw
        return out

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "TileDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
