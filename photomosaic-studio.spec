# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['D:/PyCharmProject/project/photomosaic/scripts/_entry_gui.py'],
    pathex=['D:/PyCharmProject/project/photomosaic'],
    binaries=[],
    datas=[('D:/PyCharmProject/project/photomosaic/demo_output/master.png', 'demo_output')],
    hiddenimports=['PySide6', 'PySide6.QtWidgets', 'PySide6.QtGui', 'PySide6.QtCore'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['cupy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='photomosaic-studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['D:/PyCharmProject/project/photomosaic/assets/icon.ico'],
)
