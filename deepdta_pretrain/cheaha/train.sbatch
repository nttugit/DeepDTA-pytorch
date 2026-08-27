#!/bin/bash
#SBATCH --job-name=deepdta-plm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
# Submit via submit_train.sh so --output lands in the run directory.
set -euo pipefail

: "${REPO:?REPO is required}"
: "${OUT:?OUT is required}"
: "${CMD:?CMD is required}"
: "${DATASET:?DATASET is required}"

ENV_NAME="${ENV_NAME:-deepdta-plm}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-100}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ENCODER_ARGS="${ENCODER_ARGS:-}"

# $HOME quota is small and HuggingFace caches weights there by default.
export HF_HOME="${HF_HOME:-${USER_DATA:-$HOME}/hf_cache}"
mkdir -p "$HF_HOME"

mkdir -p "$OUT"
cd "$REPO"

{
  echo "host=$(hostname)"
  echo "date=$(date -Is)"
  echo "job=${SLURM_JOB_ID:-local}"
  echo "repo=$REPO"
  echo "out=$OUT"
  echo "hf_home=$HF_HOME"
  git -C "$REPO" rev-parse HEAD 2>/dev/null || true
  git -C "$REPO" status -sb 2>/dev/null || true
} > "$OUT/job_meta.txt"

module reset
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
PY="${CONDA_PREFIX}/bin/python"

"$PY" - <<'PY'
import sys
import torch
import torch.utils
import transformers
print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
if not torch.cuda.is_available():
    sys.exit("CUDA is not available in this job")
print("gpu", torch.cuda.get_device_name(0))
PY

# Bash 4.2 (Cheaha) errors on "${empty_array[@]}" when `set -u` is on.
DATA_ARGS=(--dataset "$DATASET")
if [[ -n "${DATA_DIR:-}" ]]; then
  DATA_ARGS+=(--data-dir "$DATA_DIR")
fi

# Idempotent: loads the cache if it already exists, otherwise runs the encoders.
# Compute nodes may have no internet, so the weights must already be in $HF_HOME
# (setup_env.sh explains how to warm it from a login node).
"$PY" -m deepdta_pretrain precompute "${DATA_ARGS[@]}" ${ENCODER_ARGS:-} 2>&1 | tee "$OUT/precompute.txt"

ARGS=("${DATA_ARGS[@]}" --epochs "$EPOCHS" --out-dir "$OUT")

if [[ "$CMD" == "experiment" ]]; then
  "$PY" -m deepdta_pretrain experiment "${ARGS[@]}" ${ENCODER_ARGS:-} ${EXTRA_ARGS:-}
else
  "$PY" -m deepdta_pretrain train --fold "$FOLD" "${ARGS[@]}" ${ENCODER_ARGS:-} ${EXTRA_ARGS:-}
fi

echo "Finished. Logs: $OUT/log.txt  metrics: $OUT/metrics.json"
