# jelnyelv

CPU-first sign language recognition app (webcam) using MediaPipe for pose/hand/face detection and PyTorch for sequence classification. Runs on macOS Apple Silicon (M4 Pro), with packaged builds for macOS, Windows, and Linux.

## Prerequisites

- **Python 3.12**
- **Webcam**
- **macOS Apple Silicon** (M4 Pro or later recommended for development)
- For bootstrap on macOS: [Homebrew](https://brew.sh) (for `python@3.12`)

## Setup

### macOS (Apple Silicon)

One-command bootstrap:

```bash
./scripts/bootstrap_mac.sh
```

This will:

1. Check for Python 3.12 (install via `brew install python@3.12` if missing)
2. Create a virtual environment at `.venv`
3. Install dependencies (using `uv` if available, else `pip`)
4. Verify imports (`torch`, `cv2`, `mediapipe`, `gradio`)

### Windows / Linux

1. Ensure Python 3.12 is installed.
2. Create a venv and install:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or: .venv\Scripts\Activate.ps1  # Windows
   pip install -e ".[dev]"
   ```

3. Verify: `python -c "import torch, cv2, mediapipe, gradio; print('OK')"`

**Check which Python you're using** (avoids running the wrong interpreter):

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
```

### Reproducible install (lock file)

For reproducible installs (CI, new machine, PyInstaller build env):

```bash
pip install -r requirements.lock.txt
pip install -e ".[dev]"
```

Install the lock file first so dependency versions are fixed; then the editable install so your local package wins. To regenerate the lock file: `pip freeze > requirements.lock.txt`

## Run

```bash
make run
# or
./scripts/run_dev.sh        # macOS/Linux
.\scripts\run_dev.ps1       # Windows
# or directly
python -m jelnyelv.main
```

The Gradio app starts at `http://127.0.0.1:7860` (or next available port) and opens your default browser. Entry point for PyInstaller: `python -m jelnyelv.main`.

## Build

Produce a single-file executable for the **current platform**:

```bash
make build
```

- **macOS**: `./scripts/build_pyinstaller.sh` → `dist/Jelnyelv`
- **Windows**: `.\scripts\build_pyinstaller.ps1` → `dist\Jelnyelv.exe`
- **Linux**: `./scripts/build_pyinstaller.sh` → `dist/Jelnyelv`

The executable starts the Gradio server and opens the default browser. Build uses `Jelnyelv.spec` (collects MediaPipe, OpenCV, Gradio, ONNXRuntime assets).

### Platform-independent builds

"Platform-independent" means **separate builds per OS**. Each executable must be built on (or for) its target platform. The GitHub Actions workflow builds all three:

- `macos-latest` → `jelnyelv-macos`
- `windows-latest` → `jelnyelv-windows.exe`
- `ubuntu-latest` → `jelnyelv-linux`

Artifacts are attached to workflow runs. For releases, add a release workflow that uploads these artifacts to GitHub Releases.

## Dev commands

| Command     | Description                    |
| ----------- | ------------------------------ |
| `make setup`   | Run bootstrap (macOS)          |
| `make run`     | Start the app                  |
| `make format`  | ruff format + black            |
| `make lint`    | ruff check                     |
| `make typecheck` | mypy src                    |
| `make test`    | pytest tests                   |
| `make build`   | PyInstaller build              |

## MediaPipe Tasks models (required)

The app uses MediaPipe Tasks (not the legacy `mp.solutions`). Download the model files:

```bash
python scripts/download_mediapipe_models.py
```

This creates `assets/mediapipe/` with `pose_landmarker.task`, `face_landmarker.task`, `hand_landmarker.task`. If models are missing, you'll see a clear error when starting the app.

## Teaching new signs (workflow)

Labels come from **recorded data folders** (`data/jelek/<label>/...`), not from `szavak.txt`.

1. **Add word** (optional) – In the Record tab, type a new word in "Add new word" and click "Add word". Word is added to `szavak.txt` for planning. Duplicates are rejected.
2. **Record sequences** – Select a word (from dropdown: existing folders + szavak), then click "Record one sequence" (31 frames) or "Record all 31 sequences". Data goes to `data/jelek/<word>/<sequence>/<frame>.npy`.
3. **Train** – In the Train tab, click "Train model". Labels are derived from folders with at least one complete sequence. Saves `your_model.pth` (with labels inside) and `ertekelesi_jelentes.txt`.
4. **Recognize** – In the Recognize tab, start webcam. Labels come from the checkpoint only. If you just retrained, click "Reload model".

## Inspecting a saved checkpoint

Check what keys are in your `.pth` file:

```bash
python -c "
import torch
ckpt = torch.load('your_model.pth', map_location='cpu')
keys = list(ckpt.keys()) if hasattr(ckpt, 'keys') else 'not a dict'
print('Keys:', keys)
if isinstance(ckpt, dict) and not any(k.startswith('lstm') or k.startswith('fc') for k in ckpt.keys()):
    for k in ckpt:
        v = ckpt[k]
        print(f'  {k}: {type(v).__name__}')
"
```

## Troubleshooting

### Camera permission

- **macOS**: System Settings → Privacy & Security → Camera → enable for Terminal/iTerm (or your IDE). When running the packaged app, add the built executable (e.g. Jelnyelv) if it appears in the list.
- **macOS "camera failed to properly initialize"**: The run scripts set `OPENCV_AVFOUNDATION_SKIP_AUTH=1`. If running manually, use: `OPENCV_AVFOUNDATION_SKIP_AUTH=1 python -m jelnyelv.main`
- **Linux**: Ensure your user is in the `video` group.
- **Browser**: Allow camera access when Gradio prompts.

### MediaPipe install

If `pip install mediapipe` fails (e.g. on older Python or incompatible glibc):

- Use Python 3.12.
- On Linux, ensure `libgomp1` is installed: `sudo apt install libgomp1`.

### PyTorch on Apple Silicon

- The default `pip install torch` provides CPU + MPS (Metal) support on Apple Silicon.
- To force CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- MPS is automatically used when available for faster inference.

### Import errors

If bootstrap reports missing modules, ensure the venv is activated and run:

```bash
pip install -e ".[dev]"
```

### Model loading (`KeyError: 'model_state_dict'`)

If recognition fails with this error, the checkpoint format is incompatible. Inspect the keys (see above). Retrain the model; the current `train.py` saves a compatible checkpoint with `model_state_dict`, `labels`, and `config`. After retraining, click "Reload model" in the Recognize tab.

## Optional: macOS codesigning and notarization

See [PACKAGING.md](PACKAGING.md) for steps to codesign and notarize the macOS executable for distribution outside the App Store.

## Project layout

```
jelnyelvpy/
├── src/jelnyelv/       # Main package
├── scripts/            # Bootstrap, run, build scripts
├── tests/              # Tests
├── PACKAGING.md        # macOS codesigning and notarization
├── Jelnyelv.spec       # PyInstaller spec (MediaPipe, Gradio, etc.)
├── pyproject.toml      # Dependencies and tool config
└── requirements.lock.txt  # Pinned deps for reproducible installs
```
