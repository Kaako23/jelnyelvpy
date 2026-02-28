#!/usr/bin/env bash
# One-command bootstrap for macOS (Apple Silicon).
# Installs Python 3.12, creates venv, installs deps, verifies imports.
# Usage: ./scripts/bootstrap_mac.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

echo "==> jelnyelv bootstrap (macOS Apple Silicon)"
echo "    Project root: $PROJECT_ROOT"
echo ""

# Check for Python 3.12
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python $PYTHON_VERSION (e.g. via Homebrew: brew install python@3.12)"
    exit 1
fi

PYTHON_CMD=""
for p in python3.12 python3; do
    if command -v "$p" &>/dev/null; then
        ver=$("$p" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || true
        if [[ "$ver" == "$PYTHON_VERSION" ]]; then
            PYTHON_CMD="$p"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo "ERROR: Python $PYTHON_VERSION not found."
    echo "Install via Homebrew: brew install python@3.12"
    echo "Then ensure 'python3.12' or 'python3' points to it."
    exit 1
fi

echo "==> Using Python: $($PYTHON_CMD --version)"
echo ""

# Create venv
if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating venv at $VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "==> Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "==> Activated venv"
echo ""

# Install project and deps
echo "==> Installing dependencies..."
if command -v uv &>/dev/null; then
    uv pip install -e ".[dev]"
else
    pip install -e ".[dev]"
fi

echo ""
echo "==> Verifying Python..."
python -c "import sys; print('  executable:', sys.executable); print('  version:', sys.version)"
echo ""
echo "==> Verifying imports..."
python -c "
import sys
errors = []
try:
    import torch
    print(f'  torch: {torch.__version__}')
except ImportError as e:
    errors.append(f'torch: {e}')
try:
    import cv2
    print(f'  cv2 (opencv): ok')
except ImportError as e:
    errors.append(f'cv2: {e}')
try:
    import mediapipe
    print(f'  mediapipe: ok')
except ImportError as e:
    errors.append(f'mediapipe: {e}')
try:
    import gradio
    print(f'  gradio: {gradio.__version__}')
except ImportError as e:
    errors.append(f'gradio: {e}')
if errors:
    print('Errors:', errors)
    sys.exit(1)
print('')
print('All imports OK.')
"

echo ""
echo "==> Bootstrap complete."
echo "    Activate: source $VENV_DIR/bin/activate"
echo "    Run: make run  (or: python -m jelnyelv.main)"
echo ""
