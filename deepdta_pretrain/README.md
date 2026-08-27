# DeepDTA + Pre-trained Language Models

Thay hai CNN encoder của DeepDTA (Öztürk et al., 2018) bằng embedding từ pre-trained language model:

- Protein: **ESM-2** (`facebook/esm2_t12_35M_UR50D`) → 480-d
- Drug: **ChemBERTa** (`DeepChem/ChemBERTa-77M-MLM`) → 384-d

Hai encoder **frozen** (không train). Embedding được tính trước một lần cho mỗi drug/protein duy nhất rồi cache ra đĩa, nên vòng train chỉ đọc vector và chạy MLP head.

## Vì sao cache được

Số cặp nhiều nhưng số thực thể *duy nhất* rất ít:

| Dataset | Drug | Protein | Cặp có label |
| --- | --- | --- | --- |
| Davis | 68 | 442 | 30,056 |
| KIBA | 2,111 | 229 | 118,254 |

Chỉ cần 68 + 442 forward pass cho Davis thay vì 30,056 × 2. Cache Davis mất ~30 giây trên CPU laptop, tổng dung lượng vài MB. Sau đó một epoch train chỉ mất ~1 giây.

## Kiến trúc

```
SMILES ──► ChemBERTa (frozen) ──► mean pool ──► 384-d ──► LayerNorm ──► Linear 256 ──┐
                                                                                     ├─► concat 512
protein ─► ESM-2 (frozen) ──────► mean pool ──► 480-d ──► LayerNorm ──► Linear 256 ──┘
                                                                                     │
                                              1024 ─► Dropout ─► 1024 ─► Dropout ─► 512 ─► 1
```

- **mean pool** bỏ cả padding và special token (CLS/EOS). Nếu mean luôn cả padding thì protein ngắn bị loãng embedding.
- **LayerNorm mỗi nhánh** là cần thiết: activation của ESM-2 và ChemBERTa khác scale, concat thô sẽ để một nhánh áp đảo gradient.
- **Projection** về 256-d mỗi nhánh cân bằng hai modality; `--proj-dim 0` để tắt và concat trực tiếp 864-d.
- **Head giữ nguyên hình dạng DeepDTA gốc** (1024 → 1024 → 512 → 1) để so sánh công bằng với baseline CNN.

## So với `deepdta/`

| | `deepdta` (CNN–CNN) | `deepdta_pretrain` |
| --- | --- | --- |
| Drug encoder | Embedding 128 + 3× Conv1d + max-pool → 96-d | ChemBERTa mean-pool → 384-d, frozen |
| Protein encoder | Embedding 128 + 3× Conv1d + max-pool → 96-d | ESM-2 mean-pool → 480-d, frozen |
| Tokenize | charset thủ công (`CHARISOSMISET` 64 ký tự, `CHARPROTSET` 25 ký tự) | tokenizer HuggingFace |
| Input MLP | 192-d | 864-d (hoặc 512-d khi bật projection) |
| Head / loss / optimizer / fold / metrics | — | **không đổi**, dùng lại code `deepdta/` |
| `shuffle` khi train | `False` | `True` |

Head, `train_one_run`, `Logger`, `set_seed`, metrics (CI / MSE / rm² / AUPR) và fold splits đều import trực tiếp từ `deepdta`, không copy lại. Thay đổi duy nhất trong `deepdta/` là thêm tham số `weight_decay=0.0` vào `train_one_run` (mặc định không đổi hành vi baseline).

`shuffle=True` là khác biệt có chủ ý: với MLP trên feature tĩnh, batch không shuffle sẽ bị gom theo thứ tự drug và làm gradient lệch.

## Chạy

Từ thư mục gốc repo (`BindingAffinityPrediction-DeepDTA/`):

```bash
# 1. Build cache (một lần cho mỗi dataset + encoder + pool + max_len)
python -m deepdta_pretrain precompute --dataset davis
python -m deepdta_pretrain precompute --dataset kiba

# 2. Train một fold
python -m deepdta_pretrain train --dataset davis --fold 0
python -m deepdta_pretrain train --dataset kiba --fold 0 --eval-on test

# 3. Protocol paper: 5 train fold, cùng test set
python -m deepdta_pretrain experiment --dataset kiba

# 4. Evaluate / predict
python -m deepdta_pretrain evaluate --dataset kiba --fold 0 --split test \
  --checkpoint checkpoints/deepdta_pretrain/<run>/best.pt
python -m deepdta_pretrain predict --checkpoint checkpoints/deepdta_pretrain/<run>/best.pt \
  --smiles CCO --protein MKKFFDSRREQG
```

Bước `precompute` là tùy chọn — `train` tự build cache nếu chưa có. Nhưng tách riêng thì tốt hơn: chạy `precompute` trên máy có internet trước, rồi job GPU không cần tải model.

`predict` đọc `encoder_cfg` lưu trong checkpoint nên không cần lặp lại flag encoder.

### Flag chính

| Flag | Default | Ghi chú |
| --- | --- | --- |
| `--protein-model` | `facebook/esm2_t12_35M_UR50D` | đổi sang `facebook/esm2_t33_650M_UR50D` (1280-d) nếu có GPU |
| `--drug-model` | `DeepChem/ChemBERTa-77M-MLM` | |
| `--pool` | `mean` | `cls` cho ChemBERTa thường kém hơn vì chỉ pretrain MLM |
| `--max-prot-len` | `1022` | giới hạn của ESM-2 (`max_position_embeddings=1026`) |
| `--max-smi-len` | `512` | giới hạn của ChemBERTa |
| `--proj-dim` | `256` | `0` để concat trực tiếp |
| `--encode-device` | `auto` | device cho bước encode, tách khỏi `--device` của train |
| `--encode-batch-size` | `8` | tăng lên nếu GPU rộng |
| `--rebuild-cache` | off | tính lại kể cả khi cache đã có |

Cache nằm ở `cache/embeddings/<dataset>/<kind>_<model>_<pool>_<max_len>.npz`. Key file gồm cả tên model nên đổi checkpoint sẽ sinh cache mới, không lẫn. Mỗi file lưu kèm `keys` (id drug/protein) và bị từ chối khi load nếu thứ tự entity không còn khớp dataset.

## Cảnh báo: truncation protein

ESM-2 chỉ nhận 1022 residue nhưng dữ liệu thật dài hơn nhiều:

| Dataset | Protein dài > 1022 | Dài nhất |
| --- | --- | --- |
| Davis | 109 / 442 (25%) | 2,549 |
| KIBA | 42 / 229 (18%) | 4,128 |

Nghĩa là 1/4 protein Davis bị mất phần cuối. Số bị cắt được in ra ở `precompute` và ghi vào metadata cache. Nếu kết quả Davis kém hơn baseline CNN thì đây là nghi phạm đầu tiên — hướng xử lý là sliding-window + mean-pool các chunk thay vì cắt.

`--max-prot-len` để dạng flag chính vì lý do đó, nhưng tăng quá 1022 sẽ vượt position embedding của ESM-2.

## Metrics

Giống paper: CI, MSE, rm², AUPR (ngưỡng pKd ≥ 7 cho Davis, KIBA ≥ 12.1). Mốc CNN–CNN của paper (average 5 fold):

| Dataset | CI | MSE |
| --- | --- | --- |
| Davis | 0.878 | 0.261 |
| KIBA | 0.863 | 0.194 |

So sánh trực tiếp với baseline trong repo:

```bash
python -m deepdta train --dataset davis --fold 0            # CNN–CNN
python -m deepdta_pretrain train --dataset davis --fold 0   # ESM-2 + ChemBERTa
```

## Phụ thuộc

`torch`, `numpy`, `transformers`. `tqdm` tùy chọn.

```bash
pip install transformers
```

## Cheaha (UAB HPC)

Script trong `deepdta_pretrain/cheaha/`. Login node chỉ dùng để `cd`, `git`, `sbatch` — trừ bước warm cache HuggingFace (cần internet, compute node có thể không có).

### 1. Env (một lần)

```bash
module reset && module load Anaconda3
cd ~/src/BindingAffinityPrediction-DeepDTA
bash deepdta_pretrain/cheaha/setup_env.sh
```

Env mặc định `deepdta-plm` (torch CUDA + transformers).

### 2. Warm HuggingFace cache + build embedding cache (một lần, trên login node)

`$HOME` trên Cheaha có quota nhỏ và HuggingFace mặc định cache weight vào đó, nên phải trỏ `HF_HOME` sang `$USER_DATA`:

```bash
module reset && module load Anaconda3 && conda activate deepdta-plm
export HF_HOME="$USER_DATA/hf_cache"
cd ~/src/BindingAffinityPrediction-DeepDTA
python -m deepdta_pretrain precompute --dataset kiba --encode-device cpu
python -m deepdta_pretrain precompute --dataset davis --encode-device cpu
```

Model 35M chạy CPU vài phút là xong. `train.sbatch` cũng tự export `HF_HOME` như vậy.

### 3. Submit

```bash
# smoke: 2 epoch
bash deepdta_pretrain/cheaha/submit_train.sh --dataset kiba --fold 0 --epochs 2 --note smoke

# một fold
bash deepdta_pretrain/cheaha/submit_train.sh --dataset kiba --fold 0 --note plm
bash deepdta_pretrain/cheaha/submit_train.sh --dataset davis --fold 0 --note plm

# protocol paper 5 fold
bash deepdta_pretrain/cheaha/submit_train.sh --cmd experiment --dataset kiba \
  --partition amperenodes-medium --time 48:00:00 --note plm

# đổi encoder (flag đi tới cả precompute và train)
bash deepdta_pretrain/cheaha/submit_train.sh --dataset kiba \
  --protein-model facebook/esm2_t33_650M_UR50D --encode-batch-size 4 --note esm650m
```

Job tạo thư mục `$USER_DATA/deepdta/runs/deepdta_pretrain/<dataset>/<YYYYMMDD-HHMM>_<note>_fold<k>/` gồm `log.txt`, `metrics.json`, `precompute.txt`, `checkpoints/best.pt`, `slurm-<job>.out`, `git_sha.txt`.

Vì embedding đã cache, phần train nhẹ hơn baseline CNN rất nhiều — partition `amperenodes` 12h là quá đủ cho một fold.

### 4. Theo dõi

```bash
squeue -u $USER
tail -f $OUT/log.txt
tail -f $OUT/slurm-<jobid>.out
scancel <jobid>
```

## Citation

```
@article{lin2023esm2,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and others},
  journal={Science}, volume={379}, number={6637}, pages={1123--1130}, year={2023}
}
@article{chithrananda2020chemberta,
  title={ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction},
  author={Chithrananda, Seyone and Grand, Gabriel and Ramsundar, Bharath},
  journal={arXiv:2010.09885}, year={2020}
}
@article{ozturk2018deepdta,
  title={DeepDTA: deep drug--target binding affinity prediction},
  author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  journal={Bioinformatics}, volume={34}, number={17}, pages={i821--i829}, year={2018}
}
```
