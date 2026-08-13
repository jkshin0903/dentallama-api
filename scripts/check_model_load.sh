#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python - <<'PY'
from main import _build_inference_system

print("[CHECK] Building inference system...")
inference = _build_inference_system()
print("[OK] Model, processor, and RAG encoder loaded successfully.")

# Release CUDA memory explicitly since this script is mainly for diagnostics
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY