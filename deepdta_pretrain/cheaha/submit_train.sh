#!/usr/bin/env bash
# Submit a deepdta_pretrain/ training job on Cheaha.
#
# From the repo root:
#   bash deepdta_pretrain/cheaha/submit_train.sh
#   bash deepdta_pretrain/cheaha/submit_train.sh --dataset davis --fold 0
#   bash deepdta_pretrain/cheaha/submit_train.sh --epochs 2 --note smoke
#   bash deepdta_pretrain/cheaha/submit_train.sh --cmd experiment --partition amperenodes-medium --time 48:00:00
#   bash deepdta_pretrain/cheaha/submit_train.sh --protein-model esm2-650m --note esm650m
#   bash deepdta_pretrain/cheaha/submit_train.sh --dataset kiba --drug-fingerprint ecfp4 --note ecfp
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SBATCH="$REPO/deepdta_pretrain/cheaha/train.sbatch"

DATASET="kiba"
FOLD="0"
CMD="train"
EPOCHS="100"
NOTE="plm"
PARTITION="amperenodes"
TIME="12:00:00"
MEM="32G"
CPUS="4"
ENV_NAME="deepdta-pytorch"
DATA_DIR=""
EXTRA_ARGS=""
RUNS_ROOT=""
DRUG_MODEL=""
PROTEIN_MODEL=""
DRUG_POOL=""
PROTEIN_POOL=""
MAX_SMI_LEN=""
MAX_PROT_LEN=""
ENCODE_BATCH_SIZE=""
LONG_STRATEGY=""
DRUG_FINGERPRINT=""
POOL_HEADS=""

usage() {
  sed -n '2,11p' "$0"
  echo "Flags: --dataset --fold --cmd train|experiment --epochs --note --partition --time --mem --cpus"
  echo "       --env-name --data-dir --runs-root --extra-args"
  echo "       --drug-model --protein-model (accept preset aliases: esm2-35m|esm2-150m|esm2-650m,"
  echo "                                     chemberta-mlm|chemberta-mtr)"
  echo "       --drug-pool mean|cls|attention|mh_attention"
  echo "       --protein-pool mean|max|attention|mh_attention --pool-heads 4"
  echo "       --long-strategy truncate|window --drug-fingerprint none|ecfp4"
  echo "       --max-smi-len --max-prot-len --encode-batch-size"
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
    --drug-model) DRUG_MODEL="$2"; shift 2 ;;
    --protein-model) PROTEIN_MODEL="$2"; shift 2 ;;
    --drug-pool) DRUG_POOL="$2"; shift 2 ;;
    --protein-pool) PROTEIN_POOL="$2"; shift 2 ;;
    --max-smi-len) MAX_SMI_LEN="$2"; shift 2 ;;
    --max-prot-len) MAX_PROT_LEN="$2"; shift 2 ;;
    --encode-batch-size) ENCODE_BATCH_SIZE="$2"; shift 2 ;;
    --long-strategy) LONG_STRATEGY="$2"; shift 2 ;;
    --drug-fingerprint) DRUG_FINGERPRINT="$2"; shift 2 ;;
    --pool-heads) POOL_HEADS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
  esac
done

# Encoder flags must reach both `precompute` and `train`, so they travel apart
# from EXTRA_ARGS (which only the training command sees).
ENCODER_ARGS=""
[[ -n "$DRUG_MODEL" ]] && ENCODER_ARGS="$ENCODER_ARGS --drug-model $DRUG_MODEL"
[[ -n "$PROTEIN_MODEL" ]] && ENCODER_ARGS="$ENCODER_ARGS --protein-model $PROTEIN_MODEL"
[[ -n "$DRUG_POOL" ]] && ENCODER_ARGS="$ENCODER_ARGS --drug-pool $DRUG_POOL"
[[ -n "$PROTEIN_POOL" ]] && ENCODER_ARGS="$ENCODER_ARGS --protein-pool $PROTEIN_POOL"
[[ -n "$MAX_SMI_LEN" ]] && ENCODER_ARGS="$ENCODER_ARGS --max-smi-len $MAX_SMI_LEN"
[[ -n "$MAX_PROT_LEN" ]] && ENCODER_ARGS="$ENCODER_ARGS --max-prot-len $MAX_PROT_LEN"
[[ -n "$ENCODE_BATCH_SIZE" ]] && ENCODER_ARGS="$ENCODER_ARGS --encode-batch-size $ENCODE_BATCH_SIZE"
[[ -n "$LONG_STRATEGY" ]] && ENCODER_ARGS="$ENCODER_ARGS --long-strategy $LONG_STRATEGY"
[[ -n "$DRUG_FINGERPRINT" ]] && ENCODER_ARGS="$ENCODER_ARGS --drug-fingerprint $DRUG_FINGERPRINT"
[[ -n "$POOL_HEADS" ]] && ENCODER_ARGS="$ENCODER_ARGS --pool-heads $POOL_HEADS"

if [[ -z "$RUNS_ROOT" ]]; then
  if [[ -n "${USER_DATA:-}" ]]; then
    RUNS_ROOT="$USER_DATA/deepdta/runs"
  else
    RUNS_ROOT="$REPO/logs/deepdta_pretrain"
  fi
fi

STAMP="$(date +%Y%m%d-%H%M)"
if [[ "$CMD" == "experiment" ]]; then
  RUN_NAME="${STAMP}_${NOTE}_experiment"
else
  RUN_NAME="${STAMP}_${NOTE}_fold${FOLD}"
fi
OUT="$RUNS_ROOT/deepdta_pretrain/${DATASET}/${RUN_NAME}"
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
  --job-name="plm-${DATASET}-${CMD}" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --output="${OUT}/slurm-%j.out" \
  --error="${OUT}/slurm-%j.err" \
  --export=ALL,REPO="$REPO",OUT="$OUT",CMD="$CMD",DATASET="$DATASET",FOLD="$FOLD",EPOCHS="$EPOCHS",ENV_NAME="$ENV_NAME",DATA_DIR="$DATA_DIR",EXTRA_ARGS="$EXTRA_ARGS",ENCODER_ARGS="$ENCODER_ARGS" \
  "$SBATCH")"

echo "$JOB_ID" > "$OUT/slurm_jobid.txt"
printf "%s\t%s\tdeepdta_pretrain\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$(date +%F)" "$JOB_ID" "$DATASET" "$CMD" "$FOLD" "$EPOCHS" "$NOTE" "$OUT" >> "$INDEX"

echo "Submitted job $JOB_ID"
echo "Run dir: $OUT"
echo "Follow:  tail -f $OUT/log.txt"
echo "Slurm:   tail -f $OUT/slurm-${JOB_ID}.out"
echo "Queue:   squeue -j $JOB_ID"
