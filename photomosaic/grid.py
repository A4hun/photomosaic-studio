"""Manual quadtree grid editor model.

Mirrors the classic mosaic-tool interaction: the master image starts as a
uniform base grid; the user left-clicks a block to split it into four
(more detail) and right-clicks to merge four siblings back into one
(less detail).  The tree's leaves are the final collage cells.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Box = Tuple[int, int, int, int]
Point = Tuple[float, float]


def _orient(ax: float, ay: float, bx: float, by: float,
            cx: float, cy: float) -> float:
    """叉积方向：>0 逆时针，<0 顺时针，=0 共线。"""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax: float, ay: float, bx: float, by: float,
                px: float, py: float) -> bool:
    return (min(ax, bx) <= px <= max(ax, bx)
            and min(ay, by) <= py <= max(ay, by))


def _segments_intersect(x1: float, y1: float, x2: float, y2: float,
                        x3: float, y3: float, x4: float, y4: float) -> bool:
    """线段 (x1,y1)-(x2,y2) 与 (x3,y3)-(x4,y4) 是否相交（含端点/共线重叠）。"""
    d1 = _orient(x3, y3, x4, y4, x1, y1)
    d2 = _orient(x3, y3, x4, y4, x2, y2)
    d3 = _orient(x1, y1, x2, y2, x3, y3)
    d4 = _orient(x1, y1, x2, y2, x4, y4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) \
            and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and _on_segment(x3, y3, x4, y4, x1, y1):
        return True
    if d2 == 0 and _on_segment(x3, y3, x4, y4, x2, y2):
        return True
    if d3 == 0 and _on_segment(x1, y1, x2, y2, x3, y3):
        return True
    if d4 == 0 and _on_segment(x1, y1, x2, y2, x4, y4):
        return True
    return False


class PolygonHit:
    """自由套索的多边形命中测试（点包含 + 矩形相交），纯 Python、不依赖 Qt。

    顶点按鼠标轨迹顺序给出，首尾自动闭合；bbox 快速剔除保证整树遍历高效。
    """

    def __init__(self, points: Sequence[Point]):
        self.pts: List[Point] = [(float(x), float(y)) for x, y in points]
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        self.bbox: Tuple[float, float, float, float] = (
            min(xs), min(ys), max(xs), max(ys))

    def contains(self, x: float, y: float) -> bool:
        """射线法判定点是否在多边形内部。"""
        bx0, by0, bx1, by1 = self.bbox
        if not (bx0 <= x <= bx1 and by0 <= y <= by1):
            return False
        inside = False
        pts = self.pts
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                xcross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < xcross:
                    inside = not inside
            j = i
        return inside

    def intersects_box(self, box: Box) -> bool:
        """多边形与矩形是否相交（顶点入格 / 格角入多边形 / 边相交）。"""
        x0, y0, x1, y1 = box
        bx0, by0, bx1, by1 = self.bbox
        if x1 <= bx0 or bx1 <= x0 or y1 <= by0 or by1 <= y0:
            return False
        for px, py in self.pts:                       # 套索顶点落在格内
            if x0 <= px < x1 and y0 <= py < y1:
                return True
        corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1),
                   ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        for cx, cy in corners:                         # 格角/格心落在套索内
            if self.contains(cx, cy):
                return True
        pts = self.pts
        n = len(pts)
        box_edges = ((x0, y0, x1, y0), (x1, y0, x1, y1),
                     (x1, y1, x0, y1), (x0, y1, x0, y0))
        for i in range(n):                             # 套索边与格边相交
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            if max(ax, bx) < x0 or min(ax, bx) > x1 \
                    or max(ay, by) < y0 or min(ay, by) > y1:
                continue
            for ex0, ey0, ex1, ey1 in box_edges:
                if _segments_intersect(ax, ay, bx, by, ex0, ey0, ex1, ey1):
                    return True
        return False


class Node:
    __slots__ = ("box", "level", "children", "parent")

    def __init__(self, box: Box, level: int, parent: Optional["Node"] = None):
        self.box = box
        self.level = level
        self.children: Optional[List["Node"]] = None
        self.parent = parent

    @property
    def is_leaf(self) -> bool:
        return self.children is None


class QuadTree:
    def __init__(self, width: int, height: int, base_x: int, base_y: int,
                 max_level: int = 5, min_px: int = 16):
        self.width = int(width)
        self.height = int(height)
        self.max_level = int(max_level)
        self.min_px = int(min_px)
        base_x = max(1, int(base_x))
        base_y = max(1, int(base_y))
        self.roots: List[Node] = []
        cw = self.width / base_x
        ch = self.height / base_y
        for gy in range(base_y):
            for gx in range(base_x):
                box = (int(gx * cw), int(gy * ch), int((gx + 1) * cw), int((gy + 1) * ch))
                self.roots.append(Node(box, 1))

    # ---- queries -------------------------------------------------------
    def leaves(self) -> List[Node]:
        out: List[Node] = []
        stack = list(self.roots)
        while stack:
            n = stack.pop()
            if n.is_leaf:
                out.append(n)
            else:
                stack.extend(n.children or ())
        out.sort(key=lambda n: (n.box[1], n.box[0]))
        return out

    def leaf_boxes(self) -> List[Tuple[Box, int]]:
        return [(n.box, n.level) for n in self.leaves()]

    def find_leaf(self, x: int, y: int) -> Optional[Node]:
        for n in self.leaves():
            x0, y0, x1, y1 = n.box
            if x0 <= x < x1 and y0 <= y < y1:
                return n
        return None

    # ---- edits ---------------------------------------------------------
    def _can_split(self, n: Node) -> bool:
        x0, y0, x1, y1 = n.box
        return (
            n.is_leaf
            and n.level < self.max_level
            and (x1 - x0) >= 2 * self.min_px
            and (y1 - y0) >= 2 * self.min_px
        )

    def _split_node(self, n: Node) -> None:
        """Split one leaf into four children (internal helper)."""
        x0, y0, x1, y1 = n.box
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        n.children = [
            Node((x0, y0, mx, my), n.level + 1, n),
            Node((mx, y0, x1, my), n.level + 1, n),
            Node((x0, my, mx, y1), n.level + 1, n),
            Node((mx, my, x1, y1), n.level + 1, n),
        ]

    def split_at(self, x: int, y: int) -> bool:
        n = self.find_leaf(x, y)
        if n is None or not self._can_split(n):
            return False
        self._split_node(n)
        return True

    def _split_to(self, n: Node, target: int) -> bool:
        """Recursively split a leaf to `target` depth (bounded by size/level)."""
        if n.level >= target or not self._can_split(n):
            return False
        self._split_node(n)
        for c in n.children or ():
            self._split_to(c, target)
        return True

    def split_fully_at(self, x: int, y: int) -> bool:
        """Split the leaf under the point as deep as the tree allows."""
        n = self.find_leaf(x, y)
        if n is None:
            return False
        return self._split_to(n, self.max_level)

    def refine_region(self, box: Box, target_level: Optional[int] = None) -> bool:
        """Subdivide the area covered by `box` down to `target_level`
        (default: the tree's finest level).  Only the parts of the tree that
        actually intersect the box are split, so the refined area follows the
        user-drawn rectangle instead of whole base cells."""
        target = self.max_level if target_level is None else max(1, int(target_level))
        rx0, ry0, rx1, ry1 = box
        changed = False

        def hit(b: Box) -> bool:
            return b[0] < rx1 and rx0 < b[2] and b[1] < ry1 and ry0 < b[3]

        def rec(n: Node) -> None:
            nonlocal changed
            if n.is_leaf:
                if n.level >= target or not self._can_split(n):
                    return
                self._split_node(n)
                changed = True
            for c in n.children or ():
                if hit(c.box):
                    rec(c)

        for n in list(self.roots):
            if hit(n.box):
                rec(n)
        return changed

    # kept for backwards compatibility
    def split_region(self, box: Box, target_level: int) -> bool:
        return self.refine_region(box, target_level)

    def clear_region(self, box: Box) -> bool:
        """Cancel subdivision inside `box`: every subdivided cell whose centre
        lies in the box is merged back into one cell (top-down, so the coarsest
        such cell wins)."""
        rx0, ry0, rx1, ry1 = box
        changed = False

        def inside(b: Box) -> bool:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            return rx0 <= cx < rx1 and ry0 <= cy < ry1

        def rec(n: Node) -> None:
            nonlocal changed
            if n.is_leaf:
                return
            if inside(n.box):
                n.children = None
                changed = True
                return
            for c in n.children or ():
                rec(c)

        for r in self.roots:
            rec(r)
        return changed

    # ---- 自由套索（任意形状选区）---------------------------------------
    def refine_polygon(self, points: Sequence[Point],
                       target_level: Optional[int] = None) -> bool:
        """沿鼠标轨迹围成的多边形细分到 `target_level`（默认树的最细级）。

        与 refine_region 相同的递归策略：只细分与多边形相交的子象限，
        细分区域贴合套索边界而不是整块基础格。"""
        if len(points) < 3:
            return False
        poly = PolygonHit(points)
        target = self.max_level if target_level is None else max(1, int(target_level))
        changed = False

        def rec(n: Node) -> None:
            nonlocal changed
            if n.is_leaf:
                if n.level >= target or not self._can_split(n):
                    return
                self._split_node(n)
                changed = True
            for c in n.children or ():
                if poly.intersects_box(c.box):
                    rec(c)

        for n in list(self.roots):
            if poly.intersects_box(n.box):
                rec(n)
        return changed

    def clear_polygon(self, points: Sequence[Point]) -> bool:
        """取消套索范围内的细分（refine_polygon 的逆操作）。

        自底向上：与多边形相交的"全叶子"兄弟组合并回父块。用同一个套索
        再扫一遍即可完全撤销刚才的细分；未被套索碰到的细分子树会保留，
        并阻止其祖先被级联合并。"""
        if len(points) < 3:
            return False
        poly = PolygonHit(points)
        changed = False

        def rec(n: Node) -> None:
            nonlocal changed
            if n.is_leaf:
                return
            for c in n.children or ():
                rec(c)
            if (n.children and all(c.is_leaf for c in n.children)
                    and poly.intersects_box(n.box)):
                n.children = None
                changed = True

        for r in self.roots:
            rec(r)
        return changed

    def min_leaf_px(self) -> int:
        """当前参数下能达到的最细叶子边长（像素）：基础格边 / 2^(级数-1)。"""
        if not self.roots:
            return 0
        base = min(min(r.box[2] - r.box[0], r.box[3] - r.box[1])
                   for r in self.roots)
        px = base
        for _ in range(self.max_level - 1):
            if px < 2 * self.min_px:
                break
            px //= 2
        return max(1, px)

    def set_max_level(self, level: int) -> bool:
        """Change the finest allowed level; cells deeper than it are merged."""
        level = max(1, int(level))
        self.max_level = level
        changed = False
        for p in list(self._internal_nodes()):
            if p.children is not None and p.level >= level:
                p.children = None
                changed = True
        return changed

    def merge_up_at(self, x: int, y: int) -> bool:
        """Undo one level at the point: merge the clicked cell with its siblings
        (and anything refined inside them) back into their parent."""
        n = self.find_leaf(x, y)
        if n is None or n.parent is None:
            return False
        n.parent.children = None
        return True

    def collapse_at(self, x: int, y: int) -> bool:
        """Cancel all subdivision of the base cell containing the point."""
        n = self.find_leaf(x, y)
        if n is None:
            return False
        a = n
        while a.parent is not None:
            a = a.parent
        if a.children is None:
            return False
        a.children = None
        return True

    def merge_at(self, x: int, y: int) -> bool:
        n = self.find_leaf(x, y)
        if n is None or n.parent is None:
            return False
        p = n.parent
        if p.children is None or any(c.children for c in p.children):
            return False  # siblings must all be leaves
        p.children = None
        return True

    def merge_level(self, level: int) -> int:
        """Collapse every fully-leaf sibling group whose children are at `level`."""
        merged = 0
        for p in list(self._internal_nodes()):
            if (
                p.children
                and p.children[0].level == level
                and all(c.is_leaf for c in p.children)
            ):
                p.children = None
                merged += 4
        return merged

    def _internal_nodes(self) -> List[Node]:
        out: List[Node] = []
        stack = list(self.roots)
        while stack:
            n = stack.pop()
            if not n.is_leaf:
                out.append(n)
                stack.extend(n.children or ())
        return out

    def reset(self) -> None:
        for r in self.roots:
            r.children = None

    @classmethod
    def from_leaf_boxes(cls, width: int, height: int, base_x: int, base_y: int,
                        leaf_boxes, max_level: int = 5, min_px: int = 16) -> "QuadTree":
        """Rebuild a tree whose leaves equal `leaf_boxes` ((box, level) pairs).

        Used to restore a manually edited grid from a saved project/settings.
        """
        tree = cls(width, height, base_x, base_y, max_level=max_level, min_px=min_px)
        leafset = {tuple(b) for b, _l in leaf_boxes}

        def contains_leaf(box: Box) -> bool:
            x0, y0, x1, y1 = box
            return any(lb[0] >= x0 and lb[1] >= y0 and lb[2] <= x1 and lb[3] <= y1
                       for lb in leafset)

        def rec(node: Node) -> None:
            if tuple(node.box) in leafset:
                return
            if not contains_leaf(node.box) or not tree._can_split(node):
                return
            x0, y0, x1, y1 = node.box
            mx, my = (x0 + x1) // 2, (y0 + y1) // 2
            node.children = [
                Node((x0, y0, mx, my), node.level + 1, node),
                Node((mx, y0, x1, my), node.level + 1, node),
                Node((x0, my, mx, y1), node.level + 1, node),
                Node((mx, my, x1, y1), node.level + 1, node),
            ]
            for child in node.children:
                rec(child)

        for root in tree.roots:
            rec(root)
        return tree

    def level_histogram(self) -> dict:
        hist: dict = {}
        for n in self.leaves():
            hist[n.level] = hist.get(n.level, 0) + 1
        return dict(sorted(hist.items()))
