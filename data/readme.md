# Dataset

Place official DeepDTA files here:

```text
data/davis/   # Y, ligands_can.txt, proteins.txt, folds/
data/kiba/
data/custom/  # optional user dataset
```

Prepare with:

```bash
python scripts/download_data.py
```

Davis `Y` stores Kd in nM. Set `dataset.is_log: 1` so training uses `pKd = -log10(Kd / 1e9)`.
KIBA `Y` is already transformed; keep `is_log: 0`.
