# Binding Affinity Prediction — DeepDTA (PyTorch)

PyTorch reimplementation of [DeepDTA](https://github.com/hkmztrk/DeepDTA) (Öztürk et al., *Bioinformatics* 2018) inside the BindingAffinityPrediction project layout.

The original paper models SMILES strings and protein sequences with two 1D-CNN blocks, concatenates the representations, and regresses binding affinity with a fully connected head.

## Repository structure

```text
BindingAffinityPrediction-DeepDTA/
├── checkpoints/        # Saved model weights
├── codedebug/          # Forward-pass / overfit sanity checks
├── configs/            # YAML configs (Davis, KIBA, custom, quick)
├── data/               # Davis / KIBA / custom datasets
├── dataloader/         # Fold loading and PyTorch Dataset / DataLoader
├── evaluation/         # CI, MSE, rm2, AUPR
├── explore/            # Davis / KIBA dataset statistics
├── logs/               # Training logs and loss curves
├── models/             # DeepDTA CNN-CNN and paper baselines
├── preprocessing/      # Label / one-hot encoding and custom-data conversion
├── run/                # train / experiment / evaluate / predict
├── scripts/            # Data download and Cheaha submit scripts
└── utils.py            # Config, seed, device, checkpoint helpers
```

## Installation

```bash
conda env create -f environment.yml
conda activate deepdta
# or
pip install -r requirements.txt
```

PyTorch is required. Install a CUDA/MPS build from [pytorch.org](https://pytorch.org) if you train on GPU.

### Cheaha (UAB HPC)

Login nodes use an old `gcc` that cannot compile current Pillow. Do **not** `pip install matplotlib` there (it will try to build Pillow from source and fail with `for loop initial declarations are only allowed in C99 mode`).

```bash
# on Cheaha
cd ~/BindingAffinityPrediction-DeepDTA
bash scripts/setup_cheaha.sh
# or, inside an existing env:
conda activate deepdta-pytorch
conda install -y -c conda-forge matplotlib pillow numpy scikit-learn pyyaml tqdm
```

If you still use pip, force wheels and skip source builds:

```bash
pip install --upgrade pip wheel
pip install --only-binary=:all: "pillow>=9.5,<11" "matplotlib>=3.5,<3.10"
pip install numpy scikit-learn pyyaml tqdm
```

Training runs without matplotlib; only the loss-curve PNGs are skipped.

## Data

Davis and KIBA files from the original repository:

```bash
python scripts/download_data.py
```

If this repo sits next to the original clone at `learn/DeepDTA`, the script copies `data/davis` and `data/kiba` from there. Otherwise it downloads from [hkmztrk/DeepDTA](https://github.com/hkmztrk/DeepDTA).

| Dataset | Drugs | Proteins | Pairs | Affinity | `is_log` | SMILES / protein length |
| --- | --- | --- | --- | --- | --- | --- |
| Davis | 68 | 442 | 30,056 | Kd → pKd | 1 | 85 / 1200 |
| KIBA | 2,111 | 229 | 118,254 | KIBA score | 0 | 100 / 1000 |

Davis uses `pKd = -log10(Kd / 1e9)`. KIBA affinities in `Y` are already in the transformed form used by the paper.

```bash
python explore/dataset_stats.py
python explore/dataset_stats.py --save-plots
```

### Custom data

Place `ligands.tab`, `proteins.fasta`, and optional `Y.tab` under a folder, then:

```bash
python scripts/prepare_custom_data.py --path data/custom --split test
python run/experiment.py --config configs/custom.yaml --skip-search
```

`ligands.tab` is `id<TAB>SMILES` with a header. `proteins.fasta` is a FASTA file. `Y.tab` is a drugs × proteins matrix (`nan` for unknown pairs).

## Training

Single fold (train vs validation):

```bash
python run/train.py --config configs/davis.yaml --fold 0
python run/train.py --config configs/kiba.yaml --fold 0
```

Paper protocol (5-fold hyper-parameter search, then test):

```bash
python run/experiment.py --config configs/kiba.yaml
```

This matches the original `run_experiments.py` CLI:

```bash
python run/experiment.py \
  --config configs/kiba.yaml \
  --num-windows 32 \
  --seq-window-lengths 8 12 \
  --smi-window-lengths 4 8 \
  --batch-size 256 \
  --num-epoch 100 \
  --max-seq-len 1000 \
  --max-smi-len 100 \
  --dataset-path data/kiba/ \
  --problem-type 1 \
  --is-log 0
```

Skip the grid and use the YAML hyper-parameters:

```bash
python run/experiment.py --config configs/quick.yaml --skip-search
```

Evaluate a checkpoint:

```bash
python run/evaluate.py --config configs/kiba.yaml --checkpoint checkpoints/<run>/best.pt --split test
```

Predict one pair:

```bash
python run/predict.py --config configs/kiba.yaml --checkpoint checkpoints/<run>/last.pt \
  --smiles "CCO" --protein "MKKFFDSRREQG"
```

## Model

Default `combined_categorical` is the paper CNN–CNN model:

1. Integer-encode SMILES (64-char alphabet) and proteins (25-char alphabet), pad/truncate to a fixed length.
2. Keras-style embedding to 128-d vectors (index 0 is padding and is **not** masked, matching the original).
3. Three `Conv1d` layers with `n`, `2n`, `3n` filters (`n=32`) and ReLU, `padding=valid`.
4. Global max pooling.
5. Concatenate drug and protein vectors.
6. Dense 1024 → Dropout 0.1 → 1024 → Dropout 0.1 → 512 → 1.
7. Train with MSE and Adam (`lr=0.001`), batch size 256, 100 epochs, early stopping patience 15 on validation MSE.

Other `model.name` values from the original Keras builders and the paper ablations:

| `model.name` | Description |
| --- | --- |
| `combined_categorical` | CNN–CNN (main DeepDTA) |
| `combined_onehot` | CNN–CNN on one-hot inputs |
| `single_drug` | CNN on SMILES + protein identity |
| `single_prot` | Drug identity + CNN on protein |
| `baseline` | Identity vectors only |
| `similarity` | Pubchem drug sim + S-W protein sim → FC |
| `cnn_sw` | CNN on SMILES + S-W protein sim |
| `pubchem_cnn` | Pubchem drug sim + CNN on protein |

## Metrics

Reported metrics match the paper: concordance index (CI), MSE, `rm^2`, and AUPR after binarizing at pKd ≥ 7 (Davis) or KIBA ≥ 12.1.

Published CNN–CNN numbers (average over five training folds):

| Dataset | CI | MSE |
| --- | --- | --- |
| Davis | 0.878 | 0.261 |
| KIBA | 0.863 | 0.194 |

A PyTorch reimplementation will not bit-match Keras/TensorFlow 1.x, but the architecture, folds, loss, and training protocol are the same.

## Debug

```bash
python codedebug/overfit_one_batch.py
```

## Citation

```
@article{ozturk2018deepdta,
  title={DeepDTA: deep drug--target binding affinity prediction},
  author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i821--i829},
  year={2018},
  publisher={Oxford University Press}
}
```

Original source: https://github.com/hkmztrk/DeepDTA
