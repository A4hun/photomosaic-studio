"""程序化生成应用图标：4x4 马赛克拼块 + 角标放大镜。
输出 assets/icon.ico（多尺寸）与 assets/icon.png（256px）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)

SIZE = 512
CELL = SIZE // 4
# 经典马赛克调色板：暖色/冷色块交错
COLORS = [
    (239, 83, 80), (66, 133, 244), (52, 168, 83), (251, 188, 5),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66),
]


def draw(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / SIZE
    cell = size / 4.0
    k = 0
    for gy in range(4):
        for gx in range(4):
            c = COLORS[k % len(COLORS)]
            k += 1
            x0, y0 = gx * cell, gy * cell
            x1, y1 = x0 + cell, y0 + cell
            d.rounded_rectangle(
                [x0 + 2 * s, y0 + 2 * s, x1 - 2 * s, y1 - 2 * s],
                radius=18 * s, fill=c)
    # 放大镜（右下角）：白圈 + 手柄
    cx, cy = size * 0.72, size * 0.72
    r = size * 0.17
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, 255),
              width=int(26 * s))
    d.line([cx + r * 0.7, cy + r * 0.7, size * 0.93, size * 0.93],
           fill=(255, 255, 255, 255), width=int(34 * s))
    return img


png = draw(SIZE)
png.save(os.path.join(OUT, "icon.png"))
sizes = [16, 32, 48, 64, 128, 256]
png.save(os.path.join(OUT, "icon.ico"),
         sizes=[(s, s) for s in sizes],
         bitmap_format="png")
print("ok:", os.listdir(OUT))
