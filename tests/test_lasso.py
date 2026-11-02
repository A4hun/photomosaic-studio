"""Lasso (arbitrary-polygon) grid selection: PolygonHit + QuadTree polygon ops."""

import pytest

from photomosaic.grid import PolygonHit, QuadTree


def _tri(cx, cy, r):
    """Triangle around (cx, cy)."""
    return [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]


class TestPolygonHit:
    def test_contains_basic(self):
        poly = PolygonHit([(0, 0), (100, 0), (100, 100), (0, 100)])
        assert poly.contains(50, 50)
        assert not poly.contains(150, 50)
        assert not poly.contains(-1, 50)

    def test_contains_concave(self):
        # L-shape: the notch (upper right) is outside
        poly = PolygonHit([(0, 0), (50, 0), (50, 50), (100, 50),
                           (100, 100), (0, 100)])
        assert poly.contains(25, 25)
        assert poly.contains(75, 75)
        assert not poly.contains(75, 25)   # inside bbox, outside polygon

    def test_intersects_box(self):
        poly = PolygonHit(_tri(50, 50, 30))
        assert poly.intersects_box((40, 40, 60, 60))     # box inside polygon
        assert poly.intersects_box((0, 0, 100, 100))     # polygon inside box
        assert poly.intersects_box((45, 15, 55, 35))     # edge crossing
        assert not poly.intersects_box((200, 200, 220, 220))
        assert not poly.intersects_box((0, 0, 15, 15))   # bbox overlap but no hit


class TestQuadTreePolygon:
    def _tree(self):
        # 256x256, 4x4 base (64px cells), up to level 4 (finest 8px)
        return QuadTree(256, 256, 4, 4, max_level=4, min_px=1)

    def test_refine_polygon_only_inside(self):
        t = self._tree()
        # small triangle fully inside one base cell (cell (64..128, 64..128))
        assert t.refine_polygon(_tri(96, 96, 20))
        fine = [(b, l) for b, l in t.leaf_boxes() if l == 4]
        assert fine, "lasso should create finest-level leaves"
        # everything refined must intersect the triangle's bbox neighbourhood
        for (x0, y0, x1, y1), _l in fine:
            assert x1 > 64 and x0 < 128 and y1 > 64 and y0 < 128
        # far-away base cells untouched
        far = [l for (b, l) in t.leaf_boxes() if b[0] >= 192 and b[1] >= 192]
        assert far and all(l == 1 for l in far)

    def test_refine_polygon_respects_max_level(self):
        t = QuadTree(256, 256, 4, 4, max_level=2, min_px=1)
        t.refine_polygon(_tri(96, 96, 20))
        assert max(l for _b, l in t.leaf_boxes()) == 2

    def test_clear_polygon_undoes(self):
        t = self._tree()
        t.refine_polygon(_tri(96, 96, 20))
        assert any(l > 1 for _b, l in t.leaf_boxes())
        # clear with generous circle-ish polygon over the same area
        assert t.clear_polygon([(60, 60), (132, 60), (132, 132), (60, 132)])
        assert all(l == 1 for _b, l in t.leaf_boxes())

    def test_degenerate_polygon_rejected(self):
        t = self._tree()
        assert not t.refine_polygon([(10, 10), (20, 20)])      # <3 points
        assert not t.clear_polygon([(10, 10)])

    def test_min_leaf_px(self):
        t = self._tree()
        assert t.min_leaf_px() == 8      # 64 / 2^3
        t.set_max_level(1)
        assert t.min_leaf_px() == 64


def test_lasso_gesture_offscreen(tmp_path):
    """Simulated mouse lasso on the GridView refines only the encircled area."""
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PIL import Image
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from photomosaic.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    mp = tmp_path / "m.png"
    Image.new("RGB", (960, 1280), (90, 120, 60)).save(mp)
    w = MainWindow(load_settings=False)
    w.resize(1300, 900)
    w.master_path = str(mp)
    w._show_master()
    w._goto(2)
    w.show()
    app.processEvents()
    QTest.qWait(80)          # let the page layout settle so the view mapping is stable
    app.processEvents()
    gv = w.grid_view
    c = gv._disp.center()

    # Ctrl + left-drag a diamond-ish lasso around the centre
    path = [QPoint(c.x(), c.y() - 50), QPoint(c.x() + 50, c.y()),
            QPoint(c.x(), c.y() + 50), QPoint(c.x() - 50, c.y())]
    QTest.mousePress(gv, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ControlModifier, path[0])
    for p in path[1:]:
        QTest.mouseMove(gv, p)
        app.processEvents()
    QTest.mouseRelease(gv, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.ControlModifier, path[0])
    app.processEvents()

    levels = [l for _b, l in w.tree.leaf_boxes()]
    assert max(levels) == w.tree.max_level, "lasso must refine to finest level"
    assert levels.count(1) > 0, "cells outside the lasso stay at base level"

    # The overlay refresh after the first gesture may re-layout the view and
    # shift the widget->image mapping, so rebuild the cancel path from the
    # *current* mapping: a lasso around the refined cells' bounding box.
    app.processEvents()
    QTest.qWait(50)
    app.processEvents()
    fine = [b for b, l in w.tree.leaf_boxes() if l > 1]
    bx0 = min(b[0] for b in fine); by0 = min(b[1] for b in fine)
    bx1 = max(b[2] for b in fine); by1 = max(b[3] for b in fine)
    r = gv.img_rect_to_widget((bx0, by0, bx1, by1))
    m = 6  # margin so the lasso fully encloses the refined area
    rect_path = [QPoint(r.left() - m, r.top() - m),
                 QPoint(r.right() + m, r.top() - m),
                 QPoint(r.right() + m, r.bottom() + m),
                 QPoint(r.left() - m, r.bottom() + m)]

    # Ctrl + right-drag around the refined area cancels the refinement
    QTest.mousePress(gv, Qt.MouseButton.RightButton,
                     Qt.KeyboardModifier.ControlModifier, rect_path[0])
    for p in rect_path[1:]:
        QTest.mouseMove(gv, p)
        app.processEvents()
    QTest.mouseRelease(gv, Qt.MouseButton.RightButton,
                       Qt.KeyboardModifier.ControlModifier, rect_path[0])
    app.processEvents()
    assert all(l == 1 for _b, l in w.tree.leaf_boxes())
