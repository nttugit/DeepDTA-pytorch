# DeepDTA (PyTorch, gọn)

Bản PyTorch chỉ gồm mô hình CNN–CNN của paper (Öztürk et al., 2018). Không có ablation (one-hot, identity, similarity, single-drug/protein).

Tương đương Keras → PyTorch:

| Keras | PyTorch |
| --- | --- |
| `Embedding` | `nn.Embedding` |
| `Conv1D` (padding=valid) | `nn.Conv1d` (padding=0) |
| `GlobalMaxPooling1D` | `Tensor.amax(dim=-1)` |
| `Dense` + `Dropout` | `nn.Linear` + `nn.Dropout` |
| `Adam` + `mean_squared_error` | `torch.optim.Adam` + `nn.MSELoss` |
| `fit` / arrays | `DataLoader` + training loop |

## Chạy

Từ thư mục gốc repo (`BindingAffinityPrediction-DeepDTA/`):

```bash
python -m deepdta train --dataset kiba --fold 0
python -m deepdta train --dataset davis --fold 0 --eval-on test

python -m deepdta experiment --dataset kiba
python -m deepdta evaluate --dataset kiba --checkpoint checkpoints/deepdta/<run>/best.pt --split test
python -m deepdta predict --checkpoint checkpoints/deepdta/<run>/last.pt \
  --smiles CCO --protein MKKFFDSRREQG --dataset kiba
```

Hyper-parameter mặc định theo paper: KIBA kernel 8/12, Davis 4/8, `n=32`, batch 256, 100 epoch, early stop patience 15 trên val MSE.

Phụ thuộc: `torch`, `numpy`. `tqdm` là tùy chọn.
