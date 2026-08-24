"""Evaluate a saved DeepDTA checkpoint on a paper fold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.dataset import DeepDTARawData, make_dataloader, paper_splits
from evaluation.metrics import evaluate_all
from models.deepdta import build_model
from run.trainer import predict_loader
from utils import Logger, get_device, load_checkpoint, load_config, resolve_path, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DeepDTA checkpoint")
    parser.add_argument("--config", type=str, default="configs/kiba.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Usually checkpoints/<run>/best.pt")
    parser.add_argument("--fold", type=int, default=0, help="Which paper_splits entry (test set is shared)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_path(args.config))
    if args.device:
        cfg["device"] = args.device
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])

    ds_cfg = cfg["dataset"]
    raw = DeepDTARawData(
        dataset_path=resolve_path(ds_cfg["path"]),
        max_smi_len=ds_cfg["max_smi_len"],
        max_seq_len=ds_cfg["max_seq_len"],
        is_log=ds_cfg["is_log"],
        problem_type=ds_cfg["problem_type"],
        ligands_file=ds_cfg["ligands_file"],
        proteins_file=ds_cfg["proteins_file"],
        affinity_file=ds_cfg["affinity_file"],
        encoding="onehot" if cfg["model"]["name"] == "combined_onehot" else "label",
    )
    split = paper_splits(raw)[args.fold]
    loader = make_dataloader(
        raw,
        split[args.split],
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )
    model = build_model(
        name=cfg["model"]["name"],
        num_filters=cfg["model"]["num_filters"],
        smi_filter_length=cfg["model"]["smi_filter_length"],
        seq_filter_length=cfg["model"]["seq_filter_length"],
        n_drugs=raw.n_drugs,
        n_proteins=raw.n_proteins,
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
        fc_dims=cfg["model"]["fc_dims"],
        keras_init=False,  # load weights from checkpoint; do not re-init
        smi_vocab_size=raw.charsmiset_size + 1,
        seq_vocab_size=raw.charseqset_size + 1,
    )
    ckpt = load_checkpoint(resolve_path(args.checkpoint), map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)

    y_true, y_pred = predict_loader(model, loader, device, raw.n_drugs, raw.n_proteins)
    metrics = evaluate_all(y_true, y_pred, ds_cfg["aupr_threshold"])
    logger = Logger(resolve_path(cfg["paths"]["log_dir"]) / "eval")
    logger.log(metrics)
    out = resolve_path(cfg["paths"]["log_dir"]) / "eval" / "predictions.npz"
    np.savez(out, y_true=y_true, y_pred=y_pred)
    save_json(metrics, resolve_path(cfg["paths"]["log_dir"]) / "eval" / "metrics.json")
    print(f"Saved predictions to {out}")


if __name__ == "__main__":
    main()
