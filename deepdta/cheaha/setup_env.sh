#!/usr/bin/env bash
# Create the conda env used by deepdta/ on Cheaha.
#
# Do this on a GPU interactive job (Open OnDemand HPC Desktop / Jupyter, or
# srun --pty), not on the login node. Do not run `conda init`.
#
#   module reset
#   module load Anaconda3
#   bash deepdta/cheaha/setup_env.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-deepdta-pytorch}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. On Cheaha: module reset && module load Anaconda3" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Env '$ENV_NAME' already exists; installing/updating packages."
else
  conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"
conda install -y pytorch pytorch-cuda=12.1 numpy tqdm -c pytorch -c nvidia -c conda-forge

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda built", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not visible. If you are on a login node this is expected;")
    print("         re-run this check inside a GPU job after `conda activate`.")
PY

echo
echo "Activate later with:"
echo "  module reset && module load Anaconda3 && conda activate $ENV_NAME"
