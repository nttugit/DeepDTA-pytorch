#!/bin/bash
#SBATCH --job-name=deepdta-attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
# Submit via submit_train.sh so --output lands in the run directory.
set -euo pipefail

: "${REPO:?REPO is required}"
: "${OUT:?OUT is required}"
: "${CMD:?CMD is required}"
: "${DATASET:?DATASET is required}"

ENV_NAME="${ENV_NAME:-deepdta-pytorch}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-100}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT"
cd "$REPO"

{
  echo "host=$(hostname)"
  echo "date=$(date -Is)"
  echo "job=${SLURM_JOB_ID:-local}"
  echo "repo=$REPO"
  echo "out=$OUT"
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
print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA is not available in this job")
print("gpu", torch.cuda.get_device_name(0))
PY

# Bash 4.2 (Cheaha) errors on "${empty_array[@]}" when `set -u` is on.
ATTN_ARGS=(--attn-heads 4 --attn-dropout 0.1)
ARGS=(--dataset "$DATASET" --epochs "$EPOCHS" --out-dir "$OUT")
if [[ -n "${DATA_DIR:-}" ]]; then
  ARGS+=(--data-dir "$DATA_DIR")
fi

if [[ "$CMD" == "experiment" ]]; then
  "$PY" deepdta-attention/train.py experiment "${ARGS[@]}" "${ATTN_ARGS[@]}" ${EXTRA_ARGS:-}
else
  "$PY" deepdta-attention/train.py train --fold "$FOLD" "${ARGS[@]}" "${ATTN_ARGS[@]}" ${EXTRA_ARGS:-}
fi

echo "Finished. Logs: $OUT/log.txt metrics: $OUT/metrics.json"