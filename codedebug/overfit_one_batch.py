#!/usr/bin/env python3
"""Sanity-check encoding + a single forward/backward pass."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch.nn as nn

from dataloader.dataset import DeepDTARawData, make_dataloader
from models.deepdta import build_model
from preprocessing.encoding import encode_protein, encode_smiles
from utils import load_config, resolve_path, set_seed


def main() -> None:
    set_seed(1)
    smi = "CCO"
    seq = "MKKFFDSRREQG"
    drug = encode_smiles(smi, 100, with_label=True)
    prot = encode_protein(seq, 1000, with_label=True)
    assert drug.shape == (100,)
    assert prot.shape == (1000,)
    assert drug[0] == 42  # 'C'
    print("encoding ok", drug[:5], prot[:5])

    cfg_path = resolve_path("configs/quick.yaml")
    cfg = load_config(cfg_path)
    data_path = resolve_path(cfg["dataset"]["path"])
    if not data_path.exists():
        print(f"Dataset not found at {data_path}; encoding check passed.")
        return

    raw = DeepDTARawData(
        dataset_path=data_path,
        max_smi_len=cfg["dataset"]["max_smi_len"],
        max_seq_len=cfg["dataset"]["max_seq_len"],
        is_log=cfg["dataset"]["is_log"],
        problem_type=cfg["dataset"]["problem_type"],
    )
    loader = make_dataloader(raw, list(range(min(256, len(raw.label_row_inds)))), batch_size=32, shuffle=False)
    batch = next(iter(loader))
    model = build_model(
        name="combined_categorical",
        num_filters=32,
        smi_filter_length=8,
        seq_filter_length=12,
        n_drugs=raw.n_drugs,
        n_proteins=raw.n_proteins,
    )
    pred = model(drug=batch["drug"], protein=batch["protein"])
    loss = nn.MSELoss()(pred, batch["affinity"])
    loss.backward()
    print("forward/backward ok", float(loss.item()), pred[:3].detach().tolist())


if __name__ == "__main__":
    main()
