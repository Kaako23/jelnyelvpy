#!/usr/bin/env bash
# Run the jelnyelv app in dev mode.
# Activate venv if present, then start the Gradio app.
# Usage: ./scripts/run_dev.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-.venv}"

cd "$PROJECT_ROOT"

# macOS: avoid OpenCV camera auth from background thread (fixes "camera failed to properly initialize")
export OPENCV_AVFOUNDATION_SKIP_AUTH=1

if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
fi

exec python -m jelnyelv.main "$@"
