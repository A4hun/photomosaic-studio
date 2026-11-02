"""PyInstaller 启动入口：photomosaic 包的 GUI。

PyInstaller 直接把脚本当顶层模块跑，包内相对 import 会丢失上下文；
这里用绝对 import 转发到 photomosaic.__main__，并强制走 gui 子命令。
"""

import sys

from photomosaic.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["gui"] + sys.argv[1:]))
