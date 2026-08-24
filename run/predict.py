"""Predict binding affinity for SMILES / protein sequence pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.deepdta import build_model
from preprocessing.encoding import encode_protein, encode_smiles
from utils import get_device, load_checkpoint, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepDTA inference")
    parser.add_argument("--config", type=str, default="configs/kiba.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--smiles", type=str, required=True, help="Ligand SMILES string")
    parser.add_argument("--protein", type=str, required=True, help="Protein amino-acid sequence")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_path(args.config))
    if args.device:
        cfg["device"] = args.device
    device = get_device(cfg["device"])

    model_name = cfg["model"]["name"]
    if model_name not in {"combined_categorical", "combined_onehot"}:
        raise SystemExit("predict.py currently supports combined_categorical / combined_onehot")

    with_label = model_name != "combined_onehot"
    # Encode the same way as training (pad/truncate using this config's max lengths).
    drug = encode_smiles(args.smiles, cfg["dataset"]["max_smi_len"], with_label=with_label)
    protein = encode_protein(args.protein, cfg["dataset"]["max_seq_len"], with_label=with_label)

    model = build_model(
        name=model_name,
        num_filters=cfg["model"]["num_filters"],
        smi_filter_length=cfg["model"]["smi_filter_length"],
        seq_filter_length=cfg["model"]["seq_filter_length"],
        n_drugs=1,
        n_proteins=1,
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
        fc_dims=cfg["model"]["fc_dims"],
        keras_init=False,
    )
    ckpt = load_checkpoint(resolve_path(args.checkpoint), map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    drug_t = torch.as_tensor(drug).unsqueeze(0).to(device)
    prot_t = torch.as_tensor(protein).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(drug=drug_t, protein=prot_t).cpu().numpy().ravel()[0]
    print(f"Predicted affinity: {float(pred):.6f}")


if __name__ == "__main__":
    main()
