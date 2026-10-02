#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ ! -x .venv/bin/python ]; then
  task_python=""
  for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)' 2>/dev/null; then
      task_python="$candidate"
      break
    fi
  done
  if [ -z "$task_python" ]; then
    echo 'Install Python 3.10–3.13 with venv support; Python 3.12 is recommended.' >&2
    exit 1
  fi
  "$task_python" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), "This launcher needs Python 3.10–3.13"'
# Refuse to reuse an unrelated or incomplete environment as a finished setup.
task_fingerprint=$(sha256sum requirements.txt vendor/open-jev/pyproject.toml | sha256sum | cut -d ' ' -f 1)
if [ ! -f .venv/news-pipeline-ready ] || [ "$(cat .venv/news-pipeline-ready)" != "$task_fingerprint" ]; then
  echo 'Installing the isolated Python environment. First setup may take several minutes.'
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install 'torch==2.8.0+cu128' --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install ./vendor/open-jev
  .venv/bin/python -m pip check
  printf '%s\n' "$task_fingerprint" > .venv/news-pipeline-ready
fi
exec .venv/bin/python pipeline.py "$@"
