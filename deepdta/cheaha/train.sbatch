#!/bin/bash
#SBATCH --job-name=deepdta
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

python - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA is not available in this job")
print("gpu", torch.cuda.get_device_name(0))
PY

DATA_FLAG=()
if [[ -n "${DATA_DIR:-}" ]]; then
  DATA_FLAG=(--data-dir "$DATA_DIR")
fi

if [[ "$CMD" == "experiment" ]]; then
  python -m deepdta experiment \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --out-dir "$OUT" \
    "${DATA_FLAG[@]}" \
    $EXTRA_ARGS
else
  python -m deepdta train \
    --dataset "$DATASET" \
    --fold "$FOLD" \
    --epochs "$EPOCHS" \
    --out-dir "$OUT" \
    "${DATA_FLAG[@]}" \
    $EXTRA_ARGS
fi

echo "Finished. Logs: $OUT/log.txt  metrics: $OUT/metrics.json"
