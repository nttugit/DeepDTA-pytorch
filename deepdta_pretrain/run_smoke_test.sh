#!/usr/bin/env bash
# Local smoke test: unit checks (no GPU/data) + optional 2-epoch train if data exists.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "=== Unit: multihead_attention_pool + DeepDTAPretrain forward ==="
python3 - <<'PY'
import torch
from deepdta_pretrain.embeddings import DEFAULT_POOL_HEADS, multihead_attention_pool, mean_pool
from deepdta_pretrain.model import DeepDTAPretrain

torch.manual_seed(0)
hidden = torch.randn(4, 12, 32)
mask = torch.ones(4, 12)
special = torch.zeros(4, 12)
special[:, 0] = 1  # CLS
mask[:, -2:] = 0   # padding

attn = multihead_attention_pool(hidden, mask, special, num_heads=DEFAULT_POOL_HEADS)
mean = mean_pool(hidden, mask, special)
assert attn.shape == (4, 32)
assert not torch.allclose(attn, mean), "mh attention pool should differ from mean pool"

model = DeepDTAPretrain(drug_dim=384, prot_dim=1280, dropout=0.35)
drug = torch.randn(8, 384)
prot = torch.randn(8, 1280)
out = model(drug, prot)
assert out.shape == (8,)
print("OK: shapes, mh_attention_pool, and GELU MLP forward passed")
PY

if [[ -d "$REPO/data/davis" || -d "$REPO/deepdta/data/davis" ]]; then
  DATASET="davis"
elif [[ -d "$REPO/data/kiba" || -d "$REPO/deepdta/data/kiba" ]]; then
  DATASET="kiba"
else
  echo "=== Skip train smoke: no davis/kiba data under repo (unit tests passed) ==="
  exit 0
fi

OUT="$REPO/logs/deepdta_pretrain/smoke-$(date +%s)"
mkdir -p "$OUT"
echo "=== Train smoke: $DATASET fold 0, 2 epochs, mh_attention pool (K=4) ==="
python3 -m deepdta_pretrain train \
  --dataset "$DATASET" \
  --fold 0 \
  --epochs 2 \
  --patience 3 \
  --drug-pool mh_attention \
  --protein-pool mh_attention \
  --pool-heads 4 \
  --protein-model esm2-35m \
  --encode-batch-size 2 \
  --batch-size 64 \
  --encode-device cpu \
  --device cpu \
  --out-dir "$OUT"

echo "Smoke train finished. Logs: $OUT/log.txt"
