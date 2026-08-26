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

Luôn truyền `--out-dir` trên Cheaha (unix-timestamp mặc định khó đọc). Checkpoint nằm trong `<out-dir>/checkpoints/{best,last}.pt`.

## Cheaha (UAB HPC)

Script nằm trong `deepdta/cheaha/`. Login node (`login00x`) chỉ dùng để `cd`, `git`, `sbatch`. Train GPU phải qua Slurm.

### 0. Laptop: đẩy code lên remote

```bash
# từ máy local, trong repo này
git add deepdta
git commit -m "Add compact DeepDTA and Cheaha scripts"
git push
```

Nếu repo chưa có remote: `rsync -av --exclude .git --exclude logs --exclude checkpoints ./ BlazerID@cheaha.rc.uab.edu:~/src/BindingAffinityPrediction-DeepDTA/`

Không dùng Cursor/VS Code Remote-SSH vào login node (process sẽ bị kill). Dùng terminal SSH hoặc Open OnDemand.

### 1. SSH vào Cheaha

```bash
ssh BlazerID@cheaha.rc.uab.edu
```

Prompt phải là `[$USER@login00x ...]`. Mở https://rc.uab.edu nếu cần Duo / Open OnDemand.

### 2. Clone (một lần)

```bash
mkdir -p ~/src
cd ~/src
git clone <URL-repo> BindingAffinityPrediction-DeepDTA
cd BindingAffinityPrediction-DeepDTA
```

Lần sau: `git pull` rồi mới submit job. Đừng sửa code trên Cheaha khi job đang chạy.

### 3. Conda env trên GPU interactive job (một lần)

Không `conda init`. Không `pip install matplotlib` trên login node.

Cách dễ: Open OnDemand → **HPC Desktop** hoặc **Jupyter**, partition `amperenodes`, 1 GPU, ~1 giờ. Trong terminal compute node:

```bash
module reset
module load Anaconda3
cd ~/src/BindingAffinityPrediction-DeepDTA
bash deepdta/cheaha/setup_env.sh
```

Hoặc từ login node:

```bash
srun --pty --partition=amperenodes --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 bash
# đợi job start, prompt đổi thành c0xxx
module reset
module load Anaconda3
cd ~/src/BindingAffinityPrediction-DeepDTA
bash deepdta/cheaha/setup_env.sh
exit
```

Env mặc định: `deepdta-pytorch`. Kiểm tra in ra `cuda available True` và tên GPU.

### 4. Data Davis / KIBA (một lần)

Trên login node được (file nhỏ):

```bash
cd ~/src/BindingAffinityPrediction-DeepDTA
python3 deepdta/cheaha/download_data.py
# hoặc giữ data ngoài git tree:
# python3 deepdta/cheaha/download_data.py --dest "$USER_DATA/deepdta/data"
```

Cần có `ligands_can.txt`, `proteins.txt`, `Y`, `folds/train_fold_setting1.txt`, `folds/test_fold_setting1.txt`.

### 5. Smoke test (2 epoch) rồi train thật

```bash
cd ~/src/BindingAffinityPrediction-DeepDTA

# smoke: chắc GPU + data + env
bash deepdta/cheaha/submit_train.sh --dataset kiba --fold 0 --epochs 2 --note smoke

# một fold (paper defaults, val MSE / early stop)
bash deepdta/cheaha/submit_train.sh --dataset kiba --fold 0 --note paper
bash deepdta/cheaha/submit_train.sh --dataset davis --fold 0 --note paper

# protocol paper: 5 train folds, cùng test set (cần partition 48h)
bash deepdta/cheaha/submit_train.sh --cmd experiment --dataset kiba \
  --partition amperenodes-medium --time 48:00:00 --note paper
```

Nếu data không nằm trong `data/kiba`: thêm `--data-dir "$USER_DATA/deepdta/data/kiba"`.

Mỗi job tạo một thư mục:

`$USER_DATA/deepdta/runs/deepdta/<dataset>/<YYYYMMDD-HHMM>_<note>_fold<k>/`

Bên trong: `log.txt`, `metrics.json`, `checkpoints/best.pt`, `slurm-<job>.out`, `git_sha.txt`. Danh sách run: `$USER_DATA/deepdta/runs/index.tsv`.

### 6. Theo dõi

```bash
squeue -u $USER
tail -f $USER_DATA/deepdta/runs/deepdta/kiba/<run>/log.txt
tail -f $USER_DATA/deepdta/runs/deepdta/kiba/<run>/slurm-<jobid>.out
scancel <jobid>   # hủy nếu cần
```

`log.txt` = loss/CI từng epoch. `slurm-*.out` = CUDA OOM, import error, `sbatch` fail.

### 7. Evaluate / lấy kết quả về máy

Sau khi job xong (vẫn `sbatch` hoặc interactive GPU):

```bash
module reset && module load Anaconda3 && conda activate deepdta-pytorch
cd ~/src/BindingAffinityPrediction-DeepDTA
OUT=$USER_DATA/deepdta/runs/deepdta/kiba/<run>

python -m deepdta evaluate --dataset kiba --fold 0 --split test \
  --checkpoint "$OUT/checkpoints/best.pt"
```

Về laptop:

```bash
rsync -av BlazerID@cheaha.rc.uab.edu:'$USER_DATA/deepdta/runs/deepdta/' ./logs/deepdta-cheaha/
```

`--num-filters` / `--embed-dim` / kernel lúc evaluate phải trùng lúc train (mặc định paper thì không cần đổi).

### Partition gợi ý

| Việc | Partition | Time |
| --- | --- | --- |
| Setup env / smoke 2 epoch | `amperenodes` | 1h |
| 1 fold KIBA/Davis, 100 epoch | `amperenodes` | 12h |
| `experiment` 5 folds | `amperenodes-medium` | 48h |

1 GPU A100 là đủ. Không xin 2 GPU. P100: `--partition pascalnodes`.

Chi tiết GPU: https://docs.rc.uab.edu/cheaha/slurm/gpu/
