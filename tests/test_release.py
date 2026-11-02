"""发布加固回归测试：首次引导、崩溃兜底、版本一致、CPU 兼容消息。"""

import os
import subprocess
import sys

import pytest

from photomosaic import __version__
from photomosaic.match import resolve_backend


class TestVersionConsistency:
    def test_pyproject_matches_init(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
            text = fh.read()
        assert f'version = "{__version__}"' in text

    def test_changelog_has_current(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(root, "CHANGELOG.md"), encoding="utf-8") as fh:
            text = fh.read()
        assert f"[{__version__}]" in text


class TestBackendMessages:
    def test_cpu_always_works(self):
        xp, device, _ = resolve_backend("cpu")
        assert device == "cpu" and xp is not None

    def test_gpu_without_cupy_chinese_error(self):
        # auto 在无 cupy 时静默回退 CPU；device="gpu" 无 cupy 时给清晰中文错误
        try:
            import cupy  # noqa: F401
            pytest.skip("本机装了 CuPy，无法验证缺 cupy 的错误路径")
        except ImportError:
            pass
        with pytest.raises(RuntimeError, match="CuPy"):
            resolve_backend("gpu")

    def test_no_cupy_env_gpu_fallback(self, tmp_path):
        shim = tmp_path / "cupy.py"
        shim.write_text("raise ImportError('blocked by test')")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
        code = (
            "from photomosaic.match import resolve_backend\n"
            "xp, device, _ = resolve_backend('auto')\n"
            "print(device)\n"
        )
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, cwd=os.path.dirname(__file__),
            timeout=60)
        assert p.returncode == 0
        assert p.stdout.strip() == "cpu"


class TestGuioffscreen:
    @pytest.fixture()
    def qapp(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    def test_welcome_only_once(self, tmp_path, qapp):
        from photomosaic.gui import MainWindow

        win = MainWindow(load_settings=False)
        win._welcome_shown = True
        win._maybe_show_welcome()  # 已看过：内部直接 return，不弹窗
        win.close()

    def test_excepthook_writes_log(self, tmp_path, monkeypatch, qapp):
        from photomosaic import gui as g

        log = tmp_path / "error.log"
        monkeypatch.setattr(g, "_ERROR_LOG", str(log))
        g._install_excepthook()
        try:
            raise ValueError("boom")
        except ValueError as e:
            sys.excepthook(ValueError, e, e.__traceback__)
        assert log.exists()
        assert "boom" in log.read_text(encoding="utf-8")
        assert "photomosaic" in str(g._ERROR_LOG) or "error.log" in str(g._ERROR_LOG)

    def test_window_icon_asset(self, qapp):
        from photomosaic.gui import MainWindow

        win = MainWindow(load_settings=False)
        assert not win.windowIcon().isNull()
        win.close()
