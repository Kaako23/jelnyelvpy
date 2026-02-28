# jelnyelv – Sign Language Recognition
# Shortcuts: setup, run, format, lint, typecheck, test, build

.PHONY: setup run format lint typecheck test build

VENV ?= .venv
PYTHON ?= python

setup:
	./scripts/bootstrap_mac.sh

run:
	@export OPENCV_AVFOUNDATION_SKIP_AUTH=1; if [ -d "$(VENV)" ]; then . $(VENV)/bin/activate; fi; $(PYTHON) -m jelnyelv.main

format:
	ruff format .
	black .

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest tests -v

build:
	@case "$$(uname -s)" in \
		Darwin) ./scripts/build_pyinstaller.sh ;; \
		MINGW*|MSYS*|CYGWIN*) powershell -File scripts/build_pyinstaller.ps1 ;; \
		*) ./scripts/build_pyinstaller.sh ;; \
	esac
