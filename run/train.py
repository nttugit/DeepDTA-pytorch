"""Train a single DeepDTA split (train vs val or train vs test)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.dataset import DeepDTARawData, make_dataloader, paper_splits
from models.deepdta import build_model
from run.trainer import predict_loader, train_one_run
from evaluation.metrics import evaluate_all
from utils import (
    Logger,
    get_device,
    load_config,
    resolve_path,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepDTA (PyTorch)")
    parser.add_argument("--config", type=str, default="configs/kiba.yaml")
    parser.add_argument("--fold", type=int, default=0, help="Which of the 5 paper folds to use as val")
    parser.add_argument("--eval-on", type=str, default="val", choices=["val", "test"], help="Held-out split for this run")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--num-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-windows", type=int, default=None, help="First CNN layer filter count n")
    parser.add_argument("--smi-window-length", type=int, default=None, help="SMILES conv kernel")
    parser.add_argument("--seq-window-length", type=int, default=None, help="Protein conv kernel")
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.device:
        cfg["device"] = args.device
    if args.dataset_path:
        cfg["dataset"]["path"] = args.dataset_path
    if args.num_epoch is not None:
        cfg["train"]["epochs"] = args.num_epoch
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_windows is not None:
        cfg["model"]["num_filters"] = args.num_windows
    if args.smi_window_length is not None:
        cfg["model"]["smi_filter_length"] = args.smi_window_length
    if args.seq_window_length is not None:
        cfg["model"]["seq_filter_length"] = args.seq_window_length
    if args.model:
        cfg["model"]["name"] = args.model
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(resolve_path(args.config)), args)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])

    stamp = str(time.time())
    log_dir = resolve_path(cfg["paths"]["log_dir"]) / stamp
    ckpt_dir = resolve_path(cfg["paths"]["checkpoint_dir"]) / stamp
    fig_dir = resolve_path(cfg["paths"]["figure_dir"]) / stamp
    logger = Logger(log_dir)
    logger.log(cfg)

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
    splits = paper_splits(raw)
    split = splits[args.fold]
    eval_key = args.eval_on
    logger.log(f"Fold {args.fold}: train={len(split['train'])} {eval_key}={len(split[eval_key])}")

    train_loader = make_dataloader(
        raw,
        split["train"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"],
        num_workers=cfg["train"]["num_workers"],
    )
    eval_loader = make_dataloader(
        raw,
        split[eval_key],
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
        keras_init=cfg["model"]["keras_init"],
        smi_vocab_size=raw.charsmiset_size + 1,  # +1 = pad token 0
        seq_vocab_size=raw.charseqset_size + 1,
    )
    logger.log(model)

    result = train_one_run(
        model=model,
        train_loader=train_loader,
        val_loader=eval_loader,
        device=device,
        n_drugs=raw.n_drugs,
        n_proteins=raw.n_proteins,
        epochs=cfg["train"]["epochs"],
        learning_rate=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        patience=cfg["train"]["patience"],
        restore_best_weights=cfg["train"]["restore_best_weights"],
        aupr_threshold=ds_cfg["aupr_threshold"],
        checkpoint_path=ckpt_dir / "best.pt",  # lowest val MSE; last.pt is written beside it
        figure_path=fig_dir / f"fold{args.fold}_{eval_key}.png",
        logger=logger,
    )
    y_true, y_pred = predict_loader(model, eval_loader, device, raw.n_drugs, raw.n_proteins)
    metrics = evaluate_all(y_true, y_pred, ds_cfg["aupr_threshold"])
    logger.log(metrics)
    save_json({"config": cfg, "metrics": metrics, "history": result["history"]}, log_dir / "metrics.json")


if __name__ == "__main__":
    main()
