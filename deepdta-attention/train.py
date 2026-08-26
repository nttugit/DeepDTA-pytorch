"""Train / evaluate / predict DeepDTA-Attention with PyTorch.

CHANGE vs original ``deepdta/train.py``:
- Imports are local (this folder is not named as a valid Python package).
- Model is ``DeepDTAAttention`` (cross-attention between SMILES and protein).
- New CLI/config: ``--attn-heads``, ``--attn-dropout``. Dataset ``dummy``.
- Logs/checkpoints go under ``logs/deepdta-attention`` and ``checkpoints/deepdta-attention``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    DATASETS,
    REPO_ROOT,
    encode_protein,
    encode_smiles,
    load_dataset,
    make_loader,
    paper_splits,
)
from metrics import concordance_index, evaluate_all, mse
from model import DeepDTAAttention

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


def set_seed(seed: int = 1) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    requested = (name or "auto").lower()
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" or (
        requested == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        return torch.device(name)
    return torch.device("cpu")


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        preds.append(model(batch["drug"], batch["protein"]).detach().cpu().numpy())
        trues.append(batch["affinity"].detach().cpu().numpy())
    return np.concatenate(trues), np.concatenate(preds)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: Optional[str] = None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class Logger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "log.txt"

    def log(self, msg: Any) -> None:
        text = str(msg)
        print(text)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 15,
    checkpoint_dir: Optional[Path] = None,
    logger: Optional[Logger] = None,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {"loss": [], "val_loss": [], "ci": [], "val_ci": []}
    best_val = float("inf")
    stale = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        batch_ci = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["drug"], batch["protein"])
            loss = criterion(pred, batch["affinity"])
            loss.backward()
            optimizer.step()
            bs = batch["affinity"].size(0)
            running += loss.item() * bs
            n_seen += bs
            n_batches += 1
            batch_ci += concordance_index(
                batch["affinity"].detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
            )

        train_loss = running / max(n_seen, 1)
        train_ci = batch_ci / max(n_batches, 1)
        val_true, val_pred = predict_loader(model, val_loader, device)
        val_loss = mse(val_true, val_pred)
        val_ci = concordance_index(val_true, val_pred)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["ci"].append(train_ci)
        history["val_ci"].append(val_ci)

        msg = (
            f"Epoch {epoch:03d} | train MSE {train_loss:.5f} CI {train_ci:.5f} | "
            f"val MSE {val_loss:.5f} CI {val_ci:.5f}"
        )
        if logger is not None:
            logger.log(msg)
        else:
            print(msg)

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            if checkpoint_dir is not None:
                save_checkpoint(
                    {
                        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "epoch": epoch,
                        "val_loss": best_val,
                        "val_ci": val_ci,
                    },
                    checkpoint_dir / "best.pt",
                )
        else:
            stale += 1
            if stale >= patience:
                stopped_epoch = epoch
                if logger is not None:
                    logger.log(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    if checkpoint_dir is not None:
        save_checkpoint(
            {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": stopped_epoch,
                "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            },
            checkpoint_dir / "last.pt",
        )

    return {
        "history": history,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val,
    }


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, default="kiba", choices=["kiba", "davis", "dummy"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-filters", type=int, default=32)
    parser.add_argument("--smi-kernel", type=int, default=None)
    parser.add_argument("--seq-kernel", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Cross-attention (new). heads must divide num_filters*3; model snaps down if not.
    parser.add_argument("--attn-heads", type=int, default=4, help="Cross-attention heads (SMILES ↔ protein)")
    parser.add_argument("--attn-dropout", type=float, default=0.1, help="Dropout inside MultiheadAttention")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepDTA-Attention (CNN + cross-attention)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="Train one fold (train vs val or test)")
    _add_common(train_p)
    train_p.add_argument("--fold", type=int, default=0)
    train_p.add_argument("--eval-on", type=str, default="val", choices=["val", "test"])
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--lr", type=float, default=0.001)
    train_p.add_argument("--patience", type=int, default=15)
    train_p.add_argument("--out-dir", type=str, default=None)

    eval_p = sub.add_parser("evaluate", help="Evaluate a checkpoint on a split")
    _add_common(eval_p)
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument("--fold", type=int, default=0)
    eval_p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    pred_p = sub.add_parser("predict", help="Predict one SMILES–protein pair")
    pred_p.add_argument("--checkpoint", type=str, required=True)
    pred_p.add_argument("--smiles", type=str, required=True)
    pred_p.add_argument("--protein", type=str, required=True)
    pred_p.add_argument("--dataset", type=str, default="kiba", choices=["kiba", "davis", "dummy"])
    pred_p.add_argument("--device", type=str, default="auto")
    pred_p.add_argument("--num-filters", type=int, default=32)
    pred_p.add_argument("--smi-kernel", type=int, default=None)
    pred_p.add_argument("--seq-kernel", type=int, default=None)
    pred_p.add_argument("--embed-dim", type=int, default=128)
    pred_p.add_argument("--dropout", type=float, default=0.1)
    pred_p.add_argument("--attn-heads", type=int, default=4)
    pred_p.add_argument("--attn-dropout", type=float, default=0.1)

    exp_p = sub.add_parser("experiment", help="Paper test protocol: 5 train folds, same test set")
    _add_common(exp_p)
    exp_p.add_argument("--epochs", type=int, default=100)
    exp_p.add_argument("--lr", type=float, default=0.001)
    exp_p.add_argument("--patience", type=int, default=15)
    exp_p.add_argument("--out-dir", type=str, default=None)
    return parser


def _kernels(args: argparse.Namespace, spec: dict) -> tuple[int, int]:
    smi_k = args.smi_kernel if args.smi_kernel is not None else spec["smi_kernel"]
    seq_k = args.seq_kernel if args.seq_kernel is not None else spec["seq_kernel"]
    return smi_k, seq_k


def _make_model(args: argparse.Namespace, spec: dict) -> DeepDTAAttention:
    smi_k, seq_k = _kernels(args, spec)
    attn_heads = args.attn_heads if args.attn_heads is not None else spec.get("attn_heads", 4)
    attn_dropout = args.attn_dropout if args.attn_dropout is not None else spec.get("attn_dropout", 0.1)
    return DeepDTAAttention(
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        smi_kernel=smi_k,
        seq_kernel=seq_k,
        dropout=args.dropout,
        attn_heads=attn_heads,
        attn_dropout=attn_dropout,
    )


def _run_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = str(time.time())
    if getattr(args, "out_dir", None):
        log_dir = Path(args.out_dir)
        return log_dir, log_dir / "checkpoints"
    return REPO_ROOT / "logs" / "deepdta-attention" / stamp, REPO_ROOT / "checkpoints" / "deepdta-attention" / stamp


def cmd_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    raw, spec = load_dataset(args.dataset, args.data_dir)
    split = paper_splits(raw)[args.fold]
    log_dir, ckpt_dir = _run_dirs(args)
    logger = Logger(log_dir)
    logger.log(vars(args))
    logger.log(f"Device: {device}")
    logger.log(f"Fold {args.fold}: train={len(split['train'])} {args.eval_on}={len(split[args.eval_on])}")

    train_loader = make_loader(raw, split["train"], args.batch_size, shuffle=False, num_workers=args.num_workers)
    eval_loader = make_loader(raw, split[args.eval_on], args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = _make_model(args, spec)
    logger.log(model)
    logger.log(
        f"Cross-attention: heads={model.cross_attn.n_heads} dropout={args.attn_dropout} "
        f"(conv_dim={args.num_filters * 3})"
    )

    result = train_one_run(
        model=model,
        train_loader=train_loader,
        val_loader=eval_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_dir=ckpt_dir,
        logger=logger,
    )
    y_true, y_pred = predict_loader(model, eval_loader, device)
    metrics = evaluate_all(y_true, y_pred, spec["aupr_threshold"])
    logger.log(metrics)
    save_json({"args": vars(args), "metrics": metrics, "history": result["history"]}, log_dir / "metrics.json")


def cmd_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    raw, spec = load_dataset(args.dataset, args.data_dir)
    split = paper_splits(raw)[args.fold]
    loader = make_loader(raw, split[args.split], args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = _make_model(args, spec)
    ckpt = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    y_true, y_pred = predict_loader(model, loader, device)
    metrics = evaluate_all(y_true, y_pred, spec["aupr_threshold"])
    print(metrics)
    out = REPO_ROOT / "logs" / "deepdta-attention" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "predictions.npz", y_true=y_true, y_pred=y_pred)
    save_json(metrics, out / "metrics.json")
    print(f"Saved predictions to {out / 'predictions.npz'}")


def cmd_predict(args: argparse.Namespace) -> None:
    spec = dict(DATASETS[args.dataset])
    device = get_device(args.device)
    model = _make_model(args, spec)
    ckpt = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    drug = torch.as_tensor(encode_smiles(args.smiles, spec["max_smi_len"]), dtype=torch.long).unsqueeze(0).to(device)
    protein = torch.as_tensor(encode_protein(args.protein, spec["max_seq_len"]), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = float(model(drug, protein).cpu().numpy().ravel()[0])
    print(f"Predicted affinity: {pred:.6f}")


def cmd_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    raw, spec = load_dataset(args.dataset, args.data_dir)
    splits = paper_splits(raw)
    log_dir, ckpt_dir = _run_dirs(args)
    logger = Logger(log_dir)
    logger.log(vars(args))
    logger.log(f"Device: {device}")
    logger.log(f"Drugs={raw.n_drugs} proteins={raw.n_proteins} labeled={len(raw.label_row_inds)}")

    test_ci, test_mse = [], []
    for split in splits:
        fold = split["fold"]
        logger.log(f"Fold {fold}: train={len(split['train'])} test={len(split['test'])}")
        set_seed(args.seed)
        train_loader = make_loader(raw, split["train"], args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = make_loader(raw, split["test"], args.batch_size, shuffle=False, num_workers=args.num_workers)
        model = _make_model(args, spec)
        train_one_run(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            checkpoint_dir=ckpt_dir / f"fold{fold}",
            logger=logger,
        )
        y_true, y_pred = predict_loader(model, test_loader, device)
        metrics = evaluate_all(y_true, y_pred, spec["aupr_threshold"])
        logger.log(
            "Fold=%d CI=%.5f MSE=%.5f rm2=%.5f AUPR=%.5f"
            % (fold, metrics["ci"], metrics["mse"], metrics["rm2"], metrics["aupr"])
        )
        test_ci.append(metrics["ci"])
        test_mse.append(metrics["mse"])
        del model

    avg_ci, avg_mse = float(np.mean(test_ci)), float(np.mean(test_mse))
    std_ci = float(np.std(test_ci))
    logger.log("---FINAL RESULTS-----")
    logger.log(f"Test CI: {test_ci}")
    logger.log(f"Test MSE: {test_mse}")
    logger.log("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" % (avg_ci, avg_mse, std_ci))
    save_json(
        {"test_ci": test_ci, "test_mse": test_mse, "avg_ci": avg_ci, "avg_mse": avg_mse, "std_ci": std_ci},
        log_dir / "final.json",
    )


def main() -> None:
    args = build_parser().parse_args()
    {"train": cmd_train, "evaluate": cmd_evaluate, "predict": cmd_predict, "experiment": cmd_experiment}[args.cmd](args)


if __name__ == "__main__":
    main()
