#!/usr/bin/env bash
# Create a clean conda env for deepdta_pretrain/ on Cheaha (torch + transformers).
#
# The previous env can be a broken PyTorch install (import torch.utils fails).
# This script removes that env and recreates it. Do not run `conda init`.
#
#   module reset
#   module load Anaconda3
#   bash deepdta_pretrain/cheaha/setup_env.sh
#
# Login node is OK for this download. CUDA will show False there; the GPU job
# is the real check.
set -euo pipefail

ENV_NAME="${ENV_NAME:-deepdta-plm}"

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
python -m pip install transformers tqdm

# $HOME has a small quota on Cheaha and HuggingFace caches model weights there
# by default. Point HF_HOME at $USER_DATA for every job (see train.sbatch).
HF_HOME_DEFAULT="${USER_DATA:-$HOME}/hf_cache"
mkdir -p "$HF_HOME_DEFAULT"
echo "HuggingFace cache dir: $HF_HOME_DEFAULT"

PY="${CONDA_PREFIX}/bin/python"
HF_HOME="$HF_HOME_DEFAULT" "$PY" - <<'PY'
import sys

import torch
import torch.utils
import transformers

print("python", sys.executable)
print("torch", torch.__version__, "at", torch.__file__)
print("transformers", transformers.__version__)
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
echo "Then pre-download the encoders once (needs internet, login node is fine):"
echo "  export HF_HOME=$HF_HOME_DEFAULT"
echo "  python -m deepdta_pretrain precompute --dataset kiba --encode-device cpu"
