# DeepDTA-Attention (CNN + cross-attention)

Bản fork của `deepdta/` với **cross-attention hai chiều** giữa SMILES và protein. Chi tiết thay đổi: [CHANGES.md](CHANGES.md).

Luồng: CNN SMILES / CNN protein (giữ feature map) → mỗi token thuốc attend residue, mỗi residue attend token thuốc → global max pool → FC.

## Config mới

| Key / flag | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--attn-heads` | 4 | Số head của `nn.MultiheadAttention`. Phải chia hết `num_filters * 3` (32×3=96). Nếu không, model hạ head gần nhất. |
| `--attn-dropout` | 0.1 | Dropout trong attention |
| `--dataset dummy` | — | Bộ synthetic nhỏ trong `dummy_data/` |

KIBA/Davis giữ kernel paper (8/12 và 4/8). Dummy dùng kernel 4/4, `max_smi_len=40`, `max_seq_len=60`.

## Dummy test (nhanh, CPU)

Từ thư mục gốc repo:

```bash
python3 deepdta-attention/make_dummy_data.py

python3 deepdta-attention/train.py train \
  --dataset dummy --fold 0 --epochs 3 --batch-size 8 \
  --num-filters 8 --embed-dim 32 --attn-heads 4 \
  --patience 10 --out-dir logs/deepdta-attention/dummy-run

python3 deepdta-attention/train.py evaluate \
  --dataset dummy --fold 0 --split test --batch-size 8 \
  --num-filters 8 --embed-dim 32 \
  --checkpoint logs/deepdta-attention/dummy-run/checkpoints/best.pt

python3 deepdta-attention/train.py predict \
  --dataset dummy --num-filters 8 --embed-dim 32 \
  --checkpoint logs/deepdta-attention/dummy-run/checkpoints/last.pt \
  --smiles "CCOC1=CC=CC=C1C(=O)O" \
  --protein "MKKFFDSRREQGGSGLGSGSSGGGGSGGGYT"
```

Hoặc `bash deepdta-attention/run_dummy_test.sh`.

## Train KIBA / Davis

`--num-filters` / `--embed-dim` khi evaluate/predict phải trùng lúc train.

```bash
python3 deepdta-attention/train.py train --dataset kiba --fold 0 --attn-heads 4
python3 deepdta-attention/train.py experiment --dataset davis --attn-heads 4 --attn-dropout 0.1
```
