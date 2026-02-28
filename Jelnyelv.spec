# Jelnyelv.spec - PyInstaller spec for sign language recognition app
# Build: pyinstaller -y --clean Jelnyelv.spec
# Output: dist/Jelnyelv (onefile executable; on macOS, run ./dist/Jelnyelv)

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect datas, binaries, hiddenimports for MediaPipe, OpenCV, Gradio, ONNX
datas = []
binaries = []
hiddenimports = ["torch"]

for pkg in ["mediapipe", "cv2", "gradio", "onnxruntime"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["src/jelnyelv/main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name="Jelnyelv",
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
)
