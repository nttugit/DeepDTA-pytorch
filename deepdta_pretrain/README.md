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
Protein seq              SMILES
     │                      │
     ▼                      ▼
  ESM-2 (frozen)      ChemBERTa (frozen)
     │                      │
     ▼                      ▼
[L, 480] embeddings    [L, 384] embeddings
     │                      │
     ▼                      ▼
Mean/Max Pooling       Mean/CLS Pooling
     │                      │
     ▼                      ▼
Protein vector [480]   Ligand vector [384]
     │                      │
     ▼                      ▼
 LayerNorm(480)        LayerNorm(384)
     │                      │
     └──────────┬───────────┘
                ▼
          Concatenate [864]
                ▼
     FC(512) → ReLU → Dropout
                ▼
     FC(256) → ReLU → Dropout
                ▼
            FC(1) → Affinity
```

- **Pooling bỏ cả padding và special token** (CLS/EOS/SEP). Nếu pool luôn cả padding thì protein ngắn bị loãng embedding; `max` pool cũng mask padding về `-inf` trước khi lấy max.
- **LayerNorm ngay sau pooling, riêng cho từng nhánh**: activation của ESM-2 và ChemBERTa khác scale, concat thô sẽ để một nhánh áp đảo gradient. Chuẩn hóa riêng nên mỗi nhánh vào MLP với mean 0 / var 1 độc lập, và `weight`/`bias` học được cho phép model tự điều chỉnh lại tỉ lệ giữa hai modality.
- **Không projection** — concat giữ nguyên 864-d, đúng chiều gốc của hai PLM.
- **Head nhỏ hơn DeepDTA gốc** (512 → 256 → 1 thay vì 1024 → 1024 → 512 → 1): 0.58M tham số huấn luyện so với 2.3M. Feature đã là đại diện chất lượng cao và cố định, nên head lớn chủ yếu làm overfit.
- Chọn pooling độc lập cho từng nhánh: `--protein-pool {mean,max,attention}` và `--drug-pool {mean,cls}`. `attention` dùng self-attention pool (mean của content token làm query, softmax theo residue) — nhấn mạnh vùng binding thay vì mean pool loãng tín hiệu trên chuỗi dài.

## So với `deepdta/`

| | `deepdta` (CNN–CNN) | `deepdta_pretrain` |
| --- | --- | --- |
| Drug encoder | Embedding 128 + 3× Conv1d + max-pool → 96-d | ChemBERTa mean/CLS-pool → 384-d, frozen |
| Protein encoder | Embedding 128 + 3× Conv1d + max-pool → 96-d | ESM-2 mean/max/attention-pool → 480-d, frozen |
| Tokenize | charset thủ công (`CHARISOSMISET` 64 ký tự, `CHARPROTSET` 25 ký tự) | tokenizer HuggingFace |
| Chuẩn hóa trước concat | không | `LayerNorm` riêng cho từng nhánh |
| Input MLP | 192-d | 864-d |
| Head | 1024 → 1024 → 512 → 1 | 512 → 256 → 1 |
| Tham số huấn luyện | ~2.3M (kể cả CNN) | 0.58M |
| Loss / fold / metrics | — | **không đổi**, dùng lại code `deepdta/` |
| Early stopping chọn trên | tập test | tập val (`split["val"]`) |
| Metric báo cáo lấy từ | epoch cuối | `best.pt` |
| LR schedule | hằng số | `ReduceLROnPlateau` |
| Chuẩn hóa target | không | z-score theo train split |
| `shuffle` khi train | `False` | `True` |

`train_one_run`, `Logger`, `set_seed`, khởi tạo trọng số kiểu Keras, metrics (CI / MSE / rm² / AUPR) và fold splits đều import trực tiếp từ `deepdta`, không copy lại. Các tham số thêm vào `train_one_run` (`weight_decay`, `lr_schedule`, `grad_clip`) đều default tắt nên hành vi baseline CNN không đổi.

`shuffle=True` là khác biệt có chủ ý: với MLP trên feature tĩnh, batch không shuffle sẽ bị gom theo thứ tự drug và làm gradient lệch.

### Lưu ý khi so sánh với baseline CNN

`deepdta/cmd_experiment` vẫn truyền tập test làm `val_loader`, tức là early stopping và việc chọn `best.pt` đều nhìn thấy test (đúng như code DeepDTA gốc phát hành). `deepdta_pretrain` đã đổi sang dùng `split["val"]`. Đây là lựa chọn có ý thức, hệ quả là **`deepdta_pretrain` đang bị so ở thế bất lợi**: nó chọn model mà không hề nhìn test, còn baseline thì có. Nếu cần so tuyệt đối công bằng thì phải sửa `deepdta/` tương tự rồi chạy lại baseline.

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

`predict` đọc `encoder_cfg` lưu trong checkpoint nên không cần lặp lại flag encoder, kể cả `--long-strategy` và fingerprint.

### Nâng cấp encoder, chạy tuần tự để cô lập tác động

```bash
# Đường cơ sở (dùng lại cache có sẵn, không cần internet)
python -m deepdta_pretrain experiment --dataset kiba

# Protein lớn hơn — cần encode lại phía protein
python -m deepdta_pretrain experiment --dataset kiba --protein-model esm2-650m

# Giữ trọn protein dài thay vì cắt
python -m deepdta_pretrain experiment --dataset kiba --long-strategy window

# Vector drug tốt hơn
python -m deepdta_pretrain experiment --dataset kiba --drug-model chemberta-mtr --drug-pool cls

# Bổ sung substructure — hiệu quả nhất trên KIBA
python -m deepdta_pretrain experiment --dataset kiba --drug-fingerprint ecfp4
```

### Flag chính

Encoder:

| Flag | Default | Ghi chú |
| --- | --- | --- |
| `--protein-model` | `facebook/esm2_t12_35M_UR50D` | nhận cả alias: `esm2-35m`, `esm2-150m` (640-d), `esm2-650m` (1280-d) |
| `--drug-model` | `DeepChem/ChemBERTa-77M-MLM` | alias: `chemberta-mlm`, `chemberta-mtr` |
| `--drug-pool` | `mean` | `mean` hoặc `cls`; thử `cls` khi dùng MTR vì nó có huấn luyện qua CLS |
| `--protein-pool` | `attention` | `mean`, `max`, hoặc `attention` (self-attention pool, khuyến nghị cho protein dài) |
| `--long-strategy` | `truncate` | `window` để giữ toàn bộ protein dài, xem phần dưới |
| `--drug-fingerprint` | `none` | `ecfp4` để concat Morgan 2048-bit vào vector ChemBERTa (cần `rdkit`) |
| `--max-prot-len` | `1022` | giới hạn của ESM-2 (`max_position_embeddings=1026`) |
| `--max-smi-len` | `512` | tự clamp về 510 theo `max_position_embeddings` của ChemBERTa |
| `--encode-device` | `auto` | device cho bước encode, tách khỏi `--device` của train |
| `--encode-batch-size` | `8` | tăng lên nếu GPU rộng |
| `--rebuild-cache` | off | tính lại kể cả khi cache đã có |

Huấn luyện:

| Flag | Default | Ghi chú |
| --- | --- | --- |
| `--lr-schedule` | `plateau` | `none` \| `plateau` (factor 0.5, patience 5, min 1e-5) \| `cosine` |
| `--normalize-target` | on | z-score theo train split; `--no-normalize-target` để tắt |
| `--weight-decay` | `5e-4` | |
| `--dropout` | `0.35` | |
| `--patience` | `10` | early stopping trên val (giảm từ 15 để hạn chế overfit) |
| `--grad-clip` | `1.0` | `0` để tắt |

Cache nằm ở `cache/embeddings/<dataset>/<kind>_<model>_<pool>_<max_len>[_window].npz`, fingerprint ở `drug_ecfp4_<bits>.npz`. Key file gồm cả tên model nên đổi checkpoint sẽ sinh cache mới, không lẫn; hậu tố `_window` bị bỏ khi `truncate` để cache cũ vẫn dùng được. Mỗi file lưu kèm `keys` (id drug/protein) và bị từ chối khi load nếu thứ tự entity không còn khớp dataset.

### Chuẩn hóa target

`DeepDTAPretrain` giữ hai buffer `target_mean` / `target_std` đặt từ **chỉ train split** của mỗi fold, và `forward` trả về `out * std + mean`. Head vì thế chỉ phải hồi quy tín hiệu chuẩn hóa, còn đầu ra vẫn ở đơn vị affinity nên metric không cần đổi. Buffer nằm trong `state_dict` nên `evaluate` và `predict` nhận được tự động.

Lý do cần: Davis đặt 69.6% nhãn ở đúng một giá trị bị chặn (pKd = 5.0, tương ứng ngưỡng 10000 nM). Không ghim offset thì đầu ra trôi giữa các epoch — trong log cũ val CI đứng yên ở 0.865–0.878 nhưng val MSE nhảy 0.29 ↔ 0.56, vì CI bất biến với biến đổi đơn điệu còn MSE thì không.

## Cảnh báo: truncation protein

ESM-2 chỉ nhận 1022 residue nhưng dữ liệu thật dài hơn nhiều:

| Dataset | Protein dài > 1022 | Dài nhất |
| --- | --- | --- |
| Davis | 109 / 442 (25%) | 2,549 |
| KIBA | 42 / 229 (18%) | 4,128 |

Nghĩa là 1/4 protein Davis bị mất phần cuối. Số bị cắt được in ra ở `precompute` và ghi vào metadata cache.

`--long-strategy window` xử lý việc này: chuỗi được chia thành các cửa sổ `max_len` chồng nhau 50%, encode và pool từng cửa sổ, rồi bình quân có trọng số theo số token nội dung của cửa sổ. Protein 2549 residue của Davis thành 4 cửa sổ phủ trọn chuỗi. Chuỗi ngắn hơn `max_len` cho ra vector **bit-identical** với đường `truncate`, nên bật cờ này chỉ ảnh hưởng đúng những chuỗi thực sự bị cắt.

`--max-prot-len` vẫn để dạng flag, nhưng tăng quá 1022 sẽ vượt position embedding của ESM-2 — dùng `window` thay vì nâng giới hạn.

Phía drug, `--max-smi-len` cũng bị chặn tự động theo giới hạn thật của model. ChemBERTa là RoBERTa nên position id bị dịch thêm `pad_token_id + 1`: bảng position embedding có 515 dòng nhưng input 514 token sẽ index tới dòng 515 và crash `IndexError: index out of range in self`. `content_length_cap()` tính giới hạn này từ config và clamp xuống 510 content token, có in thông báo khi clamp. KIBA có 1/2111 SMILES dài 592 token nên chạm đúng giới hạn này (Davis dài nhất 94 token, không ảnh hưởng).

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

### Ablation

Fold 0, 25 epoch, đánh giá `best.pt` trên `split["val"]`, ESM-2 35M + ChemBERTa-MLM, CPU. Cắt ở 25 epoch nên chưa hội tụ — hai dòng cuối vẫn còn đang cải thiện ở epoch cuối.

| Cấu hình | Davis MSE | Davis CI | KIBA MSE | KIBA CI |
| --- | --- | --- | --- | --- |
| Dự đoán hằng số (variance nhãn) | 0.8005 | 0.500 | 0.7005 | 0.500 |
| Chỉ sửa đo lường (recipe cũ) | 0.4343 | 0.828 | 0.5457 | 0.685 |
| + `--lr-schedule plateau` | 0.4343 | 0.828 | 0.5457 | 0.685 |
| + normalize-target, wd 1e-4, dropout 0.2, clip | 0.3268 | 0.850 | 0.2954 | 0.790 |
| + `--long-strategy window` | 0.3384 | 0.844 | — | — |
| + `--drug-fingerprint ecfp4` | 0.3360 | 0.854 | **0.2021** | **0.845** |
| Paper CNN–CNN (5 fold, test) | 0.261 | 0.878 | 0.194 | 0.863 |

Đọc bảng này:

- **Chuẩn hóa target là thay đổi lớn nhất** ở cả hai dataset. Nó cũng giải thích tại sao dòng scheduler không nhích: trong 25 epoch, `plateau` cần 5 epoch không cải thiện mới giảm LR, mà điểm tốt nhất đã đến ở epoch 16 (Davis) và epoch 2 (KIBA) nên LR chưa kịp giảm trước khi run kết thúc. Scheduler chỉ có tác dụng khi chạy đủ 100 epoch với `patience=15`.
- **ECFP4 là mấu chốt của KIBA**, đúng như dự đoán từ cấu trúc dữ liệu: KIBA có 2111 drug / 229 protein nên phía drug gánh phần lớn tín hiệu, và vector ChemBERTa mean-pool làm mất thông tin substructure. MSE 0.2021 đã sát paper (0.194).
- **ECFP4 không giúp Davis** (0.3268 → 0.3360 MSE, dù CI nhích lên). Davis chỉ có 68 drug nên thêm 2048 chiều gần như chỉ thêm tham số để overfit. Đây là lý do cờ này tắt mặc định.
- **Windowing lại làm Davis kém đi một chút** (0.3268 → 0.3384 MSE, CI 0.850 → 0.844), dù nó khôi phục đủ 109 protein bị cắt. Cách hiểu hợp lý: kinase domain nằm ở đầu chuỗi nên phần đuôi bị `truncate` bỏ đi chủ yếu là vùng ít liên quan tới binding, và bình quân thêm 2–4 cửa sổ đuôi chỉ làm loãng vector. Nói cách khác truncation **không** phải nghi phạm như phỏng đoán ban đầu. Vẫn nên thử lại `window` khi đã đổi sang ESM-2 650M, vì model lớn hơn có thể khai thác được phần đuôi.

## Phụ thuộc

`torch`, `numpy`, `transformers`. `tqdm` tùy chọn. `rdkit` chỉ cần khi dùng `--drug-fingerprint ecfp4`.

```bash
pip install "transformers>=4.30,<5"
pip install "rdkit>=2023.3"   # chỉ khi cần ECFP4
```

### Nếu torch < 2.6

ChemBERTa trên HuggingFace chỉ có `pytorch_model.bin`, không có `model.safetensors` (ESM-2 thì có). Từ transformers 4.52.0, việc `torch.load` một file `.bin` bị chặn nếu torch < 2.6 do CVE-2025-32434:

```
ValueError: Due to a serious vulnerability issue in `torch.load` ... we now require
users to upgrade torch to at least v2.6
```

Hai cách xử lý:

- Giữ torch cũ → `pip install "transformers==4.51.3"` (bản cuối chưa có check này).
- Hoặc nâng torch >= 2.6. Lưu ý torch 2.6 không có build `cu121`, phải dùng `cu124`/`cu126`.

## Cheaha (UAB HPC)

Script trong `deepdta_pretrain/cheaha/`. Login node chỉ dùng để `cd`, `git`, `sbatch` — trừ bước warm cache HuggingFace (cần internet, compute node có thể không có).

### 1. Env (một lần)

```bash
module reset && module load Anaconda3
cd ~/src/DeepDTA-pytorch
bash deepdta_pretrain/cheaha/setup_env.sh
```

Env mặc định `deepdta-plm` (torch CUDA + transformers).

### 2. Warm HuggingFace cache + build embedding cache (một lần, trên login node)

`$HOME` trên Cheaha có quota nhỏ và HuggingFace mặc định cache weight vào đó, nên phải trỏ `HF_HOME` sang `$USER_DATA`:

```bash
module reset && module load Anaconda3 && conda activate deepdta-plm
export HF_HOME="$USER_DATA/hf_cache"
cd ~/src/DeepDTA-pytorch
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
