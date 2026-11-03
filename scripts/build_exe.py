"""Build photomosaic-studio.exe (Windows, one-file, GUI).

用法：.venv\Scripts\python scripts\build_exe.py
产物：dist\\photomosaic-studio.exe
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    dist = os.path.join(ROOT, "dist")
    build = os.path.join(ROOT, "build")
    for d in (dist, build):
        shutil.rmtree(d, ignore_errors=True)
    debug = "--console" in sys.argv
    exe = os.path.join(ROOT, ".venv", "Scripts", "pyinstaller.exe")
    if not os.path.exists(exe):
        exe = os.path.join(ROOT, ".venv", "Scripts", "pyinstaller")
    # 首次打开的演示主图：缺了就现生成，打进 exe（add-data）
    demo_master = os.path.join(ROOT, "demo_output", "master.png")
    if not os.path.exists(demo_master):
        subprocess.check_call(
            [os.path.join(ROOT, ".venv", "Scripts", "python"),
             os.path.join(ROOT, "scripts", "demo.py")], cwd=ROOT)
    name = "photomosaic-studio" + ("-debug" if debug else "")
    cmd = [
        exe,
        "--onefile",
        "--paths", ROOT,                       # 让 PyInstaller 找到 photomosaic 包
        "--console" if debug else "--windowed",
        "--name", name,
        "--icon", os.path.join(ROOT, "assets", "icon.ico"),
        "--distpath", dist,
        "--workpath", build,
        "--clean", "--noconfirm",
        # 首次打开的演示主图（PyInstaller Windows 用 ; 分隔 src;dest）
        "--add-data", demo_master + ";demo_output",
        # PySide6 在 photomosaic.gui 里条件 import，静态分析抓不到
        "--hidden-import", "PySide6",
        "--hidden-import", "PySide6.QtWidgets",
        "--hidden-import", "PySide6.QtGui",
        "--hidden-import", "PySide6.QtCore",
        # GPU 加速不打包：exe 走 CPU 匹配（CuPy 页文件 800MB+，且机器专属 CUDA）
        "--exclude-module", "cupy",
        # 入口：绝对 import 的 GUI 启动脚本（直接跑 __main__.py 会丢包上下文）
        os.path.join(ROOT, "scripts", "_entry_gui.py"),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)
    out = os.path.join(dist, name + ".exe")
    print("built:", out, os.path.getsize(out) / 1024 / 1024, "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
