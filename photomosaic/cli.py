"""Command line interface: ingest / make / run."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .db import TileDB
from .library import VARIANTS, ingest
from .render import STREAM_THRESHOLD_PX, make_mosaic


def _parse_variants(text: str) -> List[str]:
    items = [v.strip() for v in text.split(",") if v.strip()]
    bad = [v for v in items if v not in VARIANTS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown variants {bad}; choose from {sorted(VARIANTS)}"
        )
    return items or ["orig"]


def _parse_region(text: str) -> tuple:
    """'x0,y0,x1,y1[@level]' in master-image pixels; default level 2."""
    level = 2
    body = text.strip()
    if "@" in body:
        body, lvl = body.rsplit("@", 1)
        try:
            level = int(lvl)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"bad region level in {text!r}") from exc
    parts = [p for p in body.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"region must be x0,y0,x1,y1[@level], got {text!r}")
    try:
        x0, y0, x1, y1 = (int(float(p)) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"non-integer region coords in {text!r}") from exc
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), max(1, level))


def _load_regions(args) -> Optional[List[tuple]]:
    regions: List[tuple] = list(getattr(args, "region", None) or [])
    path = getattr(args, "regions_file", None)
    if path:
        import json

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data:
            if isinstance(item, dict):
                b = item["box"]
                regions.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]),
                                int(item.get("level", 2))))
            else:
                regions.append(tuple(item))
    return regions or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="photomosaic",
        description="Render photo mosaics: a master image composed of many small tile photos.",
    )
    p.add_argument("--version", action="version", version=f"photomosaic {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common_match = argparse.ArgumentParser(add_help=False)
    common_match.add_argument("--top-k", type=int, default=8,
                              help="shortlist size for near-best random pick (default 8)")
    common_match.add_argument("--radius", type=float, default=None,
                              help="max average-RGB distance for candidate prefilter")
    common_match.add_argument("--max-reuse", type=int, default=0,
                              help="max uses per tile, 0 = unlimited (default 0)")
    common_match.add_argument("--seed", type=int, default=None, help="RNG seed")
    common_match.add_argument("--cell-px", type=int, default=64,
                              help="output tile size in px when --cells-x is given (default 64)")
    common_match.add_argument("--tile-px", type=int, default=None,
                              help="override output tile size in px")
    common_match.add_argument("--levels", type=int, default=1,
                              help="adaptive subdivision depth, 1 = uniform grid (default 1)")
    common_match.add_argument("--detail-threshold", type=float, default=12.0,
                              help="detail energy above which a cell is split (default 12)")
    common_match.add_argument("--region", action="append", type=_parse_region, default=None,
                              metavar="X0,Y0,X1,Y1[@LEVEL]",
                              help="force subdivision level inside a pixel rect (repeatable)")
    common_match.add_argument("--regions-file", default=None,
                              help='JSON list: [{"box":[x0,y0,x1,y1],"level":n}, ...]')
    common_match.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                              help="matching backend: auto=cupy if available (default auto)")
    common_match.add_argument("--first-level-factor", type=float, default=1.0,
                              help="first-level cells split only above threshold*factor "
                                   "(>1 keeps big blocks whole; default 1.0)")
    common_match.add_argument("--quality", type=int, default=100, help="JPEG quality (default 100)")
    common_match.add_argument("--blend", type=float, default=0.55, metavar="0..1",
                              help="SoftLight colorization strength toward each cell's "
                                   "master color, 0 disables (default 0.55, the original "
                                   "app's 139/255)")
    common_match.add_argument("--out-width", type=int, default=None, metavar="PX",
                              help="output image width in pixels; height follows the "
                                   "master aspect (default: master size)")
    common_match.add_argument("--stream-threshold", type=int, default=None, metavar="PX",
                              help="output pixel count above which rendering switches "
                                   f"to streaming BigTIFF (default {STREAM_THRESHOLD_PX})")
    common_match.add_argument("--stream-strip-rows", type=int, default=None, metavar="ROWS",
                              help="debug/testing: force the streaming strip height in "
                                   "rows (default: auto from the memory budget)")

    pi = sub.add_parser("ingest", help="scan a folder into a tile database")
    pi.add_argument("folder")
    pi.add_argument("--db", required=True, help="SQLite tile database path")
    pi.add_argument("--ratio", type=float, default=1.0, help="cell width/height ratio (default 1)")
    pi.add_argument("--no-recursive", action="store_true")
    pi.add_argument("--variants", type=_parse_variants, default=["orig"],
                    help="comma list: orig,mirror,r90,r180,r270")
    pi.add_argument("--thumb", type=int, default=256, help="thumbnail base size (default 256)")
    pi.add_argument("--no-dedup", action="store_true", help="disable MD5 deduplication")
    pi.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes, 0 = all cores (default 0)")
    pi.add_argument("--skip-near-white", type=int, default=None,
                    help="skip tiles whose average luminance is above this (0-255)")
    pi.add_argument("--skip-near-black", type=int, default=None,
                    help="skip tiles whose average luminance is below this (0-255)")

    pm = sub.add_parser("make", help="render a mosaic using an existing tile database",
                        parents=[common_match])
    pm.add_argument("master")
    pm.add_argument("--db", required=True)
    pm.add_argument("-o", "--output", required=True)
    pm.add_argument("--cells-x", type=int, default=None,
                    help="grid width in cells (default: master width / cell-px)")
    pm.add_argument("--poster", choices=["2x3", "3x2"], default=None,
                    help="also split the result into poster pages "
                         "(saved next to the output in <output>_poster/)")

    pr = sub.add_parser("run", help="one-shot: ingest a folder into a temp db and render",
                        parents=[common_match])
    pr.add_argument("master")
    pr.add_argument("folder")
    pr.add_argument("-o", "--output", required=True)
    pr.add_argument("--db", default=None, help="keep the tile database at this path")
    pr.add_argument("--cells-x", type=int, default=None)
    pr.add_argument("--ratio", type=float, default=1.0)
    pr.add_argument("--variants", type=_parse_variants, default=["orig"])
    pr.add_argument("--thumb", type=int, default=256)

    pg = sub.add_parser("gui", help="launch the graphical interface (PySide6)")
    pg.add_argument("--master", default=None, help="pre-load this master image")
    pg.add_argument("--db", default=None, help="pre-load this tile database")
    pg.add_argument("--folder", default=None, help="pre-fill this tile folder")
    return p


def _progress_printer():
    def cb(done: int, total: int) -> None:
        pct = 100.0 * done / max(1, total)
        sys.stderr.write(f"\rmatching {done}/{total} cells ({pct:5.1f}%)")
        sys.stderr.flush()
        if done >= total:
            sys.stderr.write("\n")
    return cb


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "ingest":
        with TileDB(args.db) as db:
            stats = ingest(
                db, args.folder, ratio=args.ratio,
                recursive=not args.no_recursive, variants=args.variants,
                thumb_size=args.thumb, dedup=not args.no_dedup,
                workers=args.workers,
                skip_near_white=args.skip_near_white,
                skip_near_black=args.skip_near_black,
            )
        print(f"ingested {stats['added']} files -> {stats['rows']} tiles "
              f"(duplicates skipped: {stats['duplicates']}, unreadable: {stats['unreadable']})")
        return 0

    if args.command == "make":
        with TileDB(args.db) as db:
            stats = make_mosaic(
                args.master, db, args.output, cells_x=args.cells_x, cell_px=args.cell_px,
                tile_px=args.tile_px, top_k=args.top_k, radius=args.radius,
                max_reuse=args.max_reuse, seed=args.seed, jpeg_quality=args.quality,
                levels=args.levels, detail_threshold=args.detail_threshold,
                regions=_load_regions(args), device=args.device,
                first_level_factor=args.first_level_factor,
                stream_threshold_px=(args.stream_threshold if args.stream_threshold is not None
                                     else STREAM_THRESHOLD_PX),
                stream_strip_rows=args.stream_strip_rows,
                blend=args.blend, out_width=args.out_width,
                progress=_progress_printer(),
            )
        _print_stats(stats)
        if args.poster:
            cols, rows = (int(n) for n in args.poster.split("x"))
            _export_poster_pages(args.output, cols, rows, args.quality)
        return 0

    if args.command == "run":
        db_path = args.db
        own_db = db_path is None
        db = TileDB(db_path)
        try:
            ingest(db, args.folder, ratio=args.ratio, variants=args.variants,
                   thumb_size=args.thumb)
            print(f"tile library: {db.count()} tiles", file=sys.stderr)
            stats = make_mosaic(
                args.master, db, args.output, cells_x=args.cells_x, cell_px=args.cell_px,
                tile_px=args.tile_px, top_k=args.top_k, radius=args.radius,
                max_reuse=args.max_reuse, seed=args.seed, jpeg_quality=args.quality,
                levels=args.levels, detail_threshold=args.detail_threshold,
                regions=_load_regions(args), device=args.device,
                stream_threshold_px=(args.stream_threshold if args.stream_threshold is not None
                                     else STREAM_THRESHOLD_PX),
                stream_strip_rows=args.stream_strip_rows,
                blend=args.blend, out_width=args.out_width,
                progress=_progress_printer(),
            )
        finally:
            db.close()
            if own_db and db_path is None:
                pass  # in-memory by default via TileDB(None)
        _print_stats(stats)
        return 0

    if args.command == "gui":
        from .gui import run_gui

        return run_gui(master=args.master, db=args.db, folder=args.folder)

    return 2


def _print_stats(stats: dict) -> None:
    print(f"grid: {stats['cells_x']}x{stats['cells_y']} cells, tile size {stats['size']}")
    print(f"library: {stats['tile_count']} tiles, {stats['unique_used']} unique used")
    print(f"mean feature distance: {stats['mean_distance']:.2f}")
    print(f"variants: {stats['variant_counts']}")
    print(f"saved: {stats['output']}")


def _export_poster_pages(output: str, cols: int, rows: int, quality: int) -> None:
    import os

    from PIL import Image

    from .render import export_poster

    stem, _ = os.path.splitext(os.path.basename(output))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(output)), f"{stem}_poster")
    with Image.open(output) as im:
        im.load()
        paths = export_poster(im.convert("RGB"), cols, rows, out_dir, stem,
                              jpeg_quality=quality)
    print(f"poster pages ({cols}x{rows}): {len(paths)} files -> {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
