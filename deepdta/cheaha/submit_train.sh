#!/usr/bin/env bash
# Submit a deepdta/ training job on Cheaha.
#
# From the repo root:
#   bash deepdta/cheaha/submit_train.sh
#   bash deepdta/cheaha/submit_train.sh --dataset davis --fold 0
#   bash deepdta/cheaha/submit_train.sh --epochs 2 --note smoke
#   bash deepdta/cheaha/submit_train.sh --cmd experiment --partition amperenodes-medium --time 48:00:00
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SBATCH="$REPO/deepdta/cheaha/train.sbatch"

DATASET="kiba"
FOLD="0"
CMD="train"
EPOCHS="100"
NOTE="paper"
PARTITION="amperenodes"
TIME="12:00:00"
MEM="16G"
CPUS="4"
ENV_NAME="deepdta-pytorch"
DATA_DIR=""
EXTRA_ARGS=""
RUNS_ROOT=""

usage() {
  sed -n '2,9p' "$0"
  echo "Flags: --dataset --fold --cmd train|experiment --epochs --note --partition --time --mem --cpus"
  echo "       --env-name --data-dir --runs-root --extra-args"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --cmd) CMD="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --note) NOTE="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$RUNS_ROOT" ]]; then
  if [[ -n "${USER_DATA:-}" ]]; then
    RUNS_ROOT="$USER_DATA/deepdta/runs"
  else
    RUNS_ROOT="$REPO/logs/deepdta"
  fi
fi

STAMP="$(date +%Y%m%d-%H%M)"
if [[ "$CMD" == "experiment" ]]; then
  RUN_NAME="${STAMP}_${NOTE}_experiment"
else
  RUN_NAME="${STAMP}_${NOTE}_fold${FOLD}"
fi
OUT="$RUNS_ROOT/deepdta/${DATASET}/${RUN_NAME}"
mkdir -p "$OUT"

if git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$REPO" rev-parse HEAD > "$OUT/git_sha.txt"
  git -C "$REPO" status -sb > "$OUT/git_status.txt"
fi

INDEX="$RUNS_ROOT/index.tsv"
if [[ ! -f "$INDEX" ]]; then
  printf "date\tjob\tmodel\tdataset\tcmd\tfold\tepochs\tnote\tout\n" > "$INDEX"
fi

JOB_ID="$(sbatch --parsable \
  --job-name="dta-${DATASET}-${CMD}" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --output="${OUT}/slurm-%j.out" \
  --error="${OUT}/slurm-%j.err" \
  --export=ALL,REPO="$REPO",OUT="$OUT",CMD="$CMD",DATASET="$DATASET",FOLD="$FOLD",EPOCHS="$EPOCHS",ENV_NAME="$ENV_NAME",DATA_DIR="$DATA_DIR",EXTRA_ARGS="$EXTRA_ARGS" \
  "$SBATCH")"

echo "$JOB_ID" > "$OUT/slurm_jobid.txt"
printf "%s\t%s\tdeepdta\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$(date +%F)" "$JOB_ID" "$DATASET" "$CMD" "$FOLD" "$EPOCHS" "$NOTE" "$OUT" >> "$INDEX"

echo "Submitted job $JOB_ID"
echo "Run dir: $OUT"
echo "Follow:  tail -f $OUT/log.txt"
echo "Slurm:   tail -f $OUT/slurm-${JOB_ID}.out"
echo "Queue:   squeue -j $JOB_ID"
