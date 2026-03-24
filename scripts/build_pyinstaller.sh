#!/usr/bin/env bash
# Build a folder distribution with PyInstaller (onedir; avoids one-file size limits).
# The executable starts the Gradio server and opens the default browser.
# Usage: ./scripts/build_pyinstaller.sh
# Output: dist/Jelnyelv/Jelnyelv (run from that directory or add to PATH)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-.venv}"

cd "$PROJECT_ROOT"

if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
fi

# Ensure PyInstaller is available
pip install -e ".[dev]" 2>/dev/null || true

echo "==> Building Jelnyelv executable"
rm -rf build dist

pyinstaller -y --clean Jelnyelv.spec

echo ""
echo "==> Build complete. Run: ./dist/Jelnyelv/Jelnyelv"
echo ""
