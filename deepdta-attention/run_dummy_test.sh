#!/usr/bin/env bash
# Smoke-test DeepDTA-Attention on dummy_data (CPU, a few seconds).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 deepdta-attention/make_dummy_data.py

OUT="logs/deepdta-attention/dummy-run"
rm -rf "$OUT"

python3 deepdta-attention/train.py train \
  --dataset dummy \
  --fold 0 \
  --epochs 3 \
  --batch-size 8 \
  --num-filters 8 \
  --embed-dim 32 \
  --attn-heads 4 \
  --attn-dropout 0.1 \
  --patience 10 \
  --device cpu \
  --out-dir "$OUT"

python3 deepdta-attention/train.py evaluate \
  --dataset dummy \
  --fold 0 \
  --split test \
  --batch-size 8 \
  --num-filters 8 \
  --embed-dim 32 \
  --attn-heads 4 \
  --device cpu \
  --checkpoint "$OUT/checkpoints/best.pt"

python3 deepdta-attention/train.py predict \
  --dataset dummy \
  --num-filters 8 \
  --embed-dim 32 \
  --attn-heads 4 \
  --device cpu \
  --checkpoint "$OUT/checkpoints/last.pt" \
  --smiles "CCOC1=CC=CC=C1C(=O)O" \
  --protein "MKKFFDSRREQGGSGLGSGSSGGGGSGGGYT"

echo "OK. Logs: $OUT  checkpoints: $OUT/checkpoints"
