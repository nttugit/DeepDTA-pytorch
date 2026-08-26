#!/usr/bin/env bash
# Create a clean conda env for deepdta/ on Cheaha.
#
# The previous env can be a broken PyTorch install (import torch.utils fails).
# This script removes that env and recreates it. Do not run `conda init`.
#
#   module reset
#   module load Anaconda3
#   bash deepdta/cheaha/setup_env.sh
#
# Login node is OK for this download. CUDA will show False there; the GPU job
# is the real check.
set -euo pipefail

ENV_NAME="${ENV_NAME:-deepdta-pytorch}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. On Cheaha: module reset && module load Anaconda3" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo "Removing env '$ENV_NAME' if it exists (broken torch.utils installs are common)."
conda deactivate >/dev/null 2>&1 || true
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true

# Create a minimal env, then install official CUDA wheels. Mixing conda
# pytorch + nvidia + conda-forge is what produced the incomplete torch.
conda create -y -n "$ENV_NAME" python=3.10 pip numpy
conda activate "$ENV_NAME"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install tqdm

PY="${CONDA_PREFIX}/bin/python"
"$PY" - <<'PY'
import torch
import torch.utils
print("python", __import__("sys").executable)
print("torch", torch.__version__, "at", torch.__file__)
print("cuda built", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
else:
    print("NOTE: CUDA not visible (expected on a login node).")
PY

echo
echo "Activate later with:"
echo "  module reset && module load Anaconda3 && conda activate $ENV_NAME"
echo "Then:  python -c 'import torch, torch.utils; print(torch.__version__, torch.cuda.is_available())'"
