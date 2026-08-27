"""Train / evaluate / predict DeepDTA with frozen ESM-2 + ChemBERTa embeddings.

The training loop, metrics, logger and fold protocol are reused from
``deepdta`` so only the representation differs from the CNN–CNN baseline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepdta.metrics import evaluate_all
from deepdta.train import (
    Logger,
    get_device,
    load_checkpoint,
    predict_loader,
    save_checkpoint,
    save_json,
    set_seed,
    train_one_run,
)
from deepdta_pretrain.data import (
    DATASETS,
    REPO_ROOT,
    load_dataset,
    make_loader,
    paper_splits,
)
from deepdta_pretrain.embeddings import ENCODERS, encode_texts, encoder_defaults
from deepdta_pretrain.model import DeepDTAPretrain


def _add_encoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, default="kiba", choices=sorted(DATASETS))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--drug-model", type=str, default=ENCODERS["drug"]["hf_name"])
    parser.add_argument("--protein-model", type=str, default=ENCODERS["protein"]["hf_name"])
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--max-smi-len", type=int, default=ENCODERS["drug"]["max_len"])
    parser.add_argument("--max-prot-len", type=int, default=ENCODERS["protein"]["max_len"])
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--encode-device", type=str, default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")


def _add_common(parser: argparse.ArgumentParser) -> None:
    _add_encoder_args(parser)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--proj-dim", type=int, default=256, help="0 disables the projection")
    parser.add_argument("--dropout", type=float, default=0.1)


def _add_optim(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepDTA with pre-trained language models")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pre_p = sub.add_parser("precompute", help="Build the frozen embedding cache only")
    _add_encoder_args(pre_p)

    train_p = sub.add_parser("train", help="Train one fold (train vs val or test)")
    _add_common(train_p)
    _add_optim(train_p)
    train_p.add_argument("--fold", type=int, default=0)
    train_p.add_argument("--eval-on", type=str, default="val", choices=["val", "test"])

    eval_p = sub.add_parser("evaluate", help="Evaluate a checkpoint on a split")
    _add_common(eval_p)
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument("--fold", type=int, default=0)
    eval_p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    pred_p = sub.add_parser("predict", help="Predict one SMILES–protein pair")
    pred_p.add_argument("--checkpoint", type=str, required=True)
    pred_p.add_argument("--smiles", type=str, required=True)
    pred_p.add_argument("--protein", type=str, required=True)
    pred_p.add_argument("--device", type=str, default="auto")
    pred_p.add_argument("--drug-model", type=str, default=None)
    pred_p.add_argument("--protein-model", type=str, default=None)
    pred_p.add_argument("--pool", type=str, default=None, choices=["mean", "cls"])
    pred_p.add_argument("--max-smi-len", type=int, default=None)
    pred_p.add_argument("--max-prot-len", type=int, default=None)

    exp_p = sub.add_parser("experiment", help="Paper test protocol: 5 train folds, same test set")
    _add_common(exp_p)
    _add_optim(exp_p)
    return parser


def _load_dataset(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    return load_dataset(
        args.dataset,
        args.data_dir,
        drug_model=args.drug_model,
        protein_model=args.protein_model,
        pool=args.pool,
        max_smi_len=args.max_smi_len,
        max_prot_len=args.max_prot_len,
        device=args.encode_device,
        encode_batch_size=args.encode_batch_size,
        rebuild_cache=args.rebuild_cache,
    )


def _make_model(args: argparse.Namespace, spec: dict[str, Any]) -> DeepDTAPretrain:
    return DeepDTAPretrain(
        drug_dim=spec["drug_dim"],
        prot_dim=spec["protein_dim"],
        proj_dim=args.proj_dim,
        dropout=args.dropout,
    )


def _model_cfg(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "drug_dim": spec["drug_dim"],
        "prot_dim": spec["protein_dim"],
        "proj_dim": args.proj_dim,
        "dropout": args.dropout,
    }


def _run_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = str(time.time())
    if getattr(args, "out_dir", None):
        log_dir = Path(args.out_dir)
        return log_dir, log_dir / "checkpoints"
    return (
        REPO_ROOT / "logs" / "deepdta_pretrain" / stamp,
        REPO_ROOT / "checkpoints" / "deepdta_pretrain" / stamp,
    )


def _annotate_checkpoints(checkpoint_dir: Path, extra: dict[str, Any]) -> None:
    """Record which encoders produced the features a checkpoint was trained on."""
    for name in ("best.pt", "last.pt"):
        path = checkpoint_dir / name
        if not path.exists():
            continue
        state = load_checkpoint(path, map_location="cpu")
        state.update(extra)
        save_checkpoint(state, path)


def _restore_model(
    args: argparse.Namespace,
    spec: dict[str, Any],
    checkpoint: dict[str, Any],
) -> DeepDTAPretrain:
    cfg = dict(_model_cfg(args, spec))
    cfg.update(checkpoint.get("model_cfg") or {})
    model = DeepDTAPretrain(
        drug_dim=cfg["drug_dim"],
        prot_dim=cfg["prot_dim"],
        proj_dim=cfg["proj_dim"],
        dropout=cfg["dropout"],
    )
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model


def _warn_encoder_mismatch(
    checkpoint: dict[str, Any],
    current: dict[str, Any],
    log: Any = print,
) -> None:
    saved = checkpoint.get("encoder_cfg")
    if not saved:
        return
    for kind in ("drug", "protein"):
        for key in ("hf_name", "pool", "max_len"):
            was = (saved.get(kind) or {}).get(key)
            now = (current.get(kind) or {}).get(key)
            if was is not None and was != now:
                log(f"WARNING: {kind} {key} was {was!r} at train time, now {now!r}")


def cmd_precompute(args: argparse.Namespace) -> None:
    raw, spec = _load_dataset(args)
    print(f"Dataset {args.dataset}: drugs={raw.n_drugs} proteins={raw.n_proteins}")
    for kind, meta in raw.encoder_cfg().items():
        print(
            f"{kind}: {meta['hf_name']} dim={meta['dim']} pool={meta['pool']} "
            f"max_len={meta['max_len']} truncated={meta['n_truncated']}/{meta['n_texts']} "
            f"token_len(min/mean/max)={meta['token_len_min']}/"
            f"{meta['token_len_mean']:.1f}/{meta['token_len_max']}"
        )
    print(f"MLP input dim: {spec['drug_dim']} + {spec['protein_dim']} = "
          f"{spec['drug_dim'] + spec['protein_dim']}")


def cmd_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    raw, spec = _load_dataset(args)
    split = paper_splits(raw)[args.fold]
    log_dir, ckpt_dir = _run_dirs(args)
    logger = Logger(log_dir)
    logger.log(vars(args))
    logger.log(f"Device: {device}")
    logger.log(f"Encoders: {raw.encoder_cfg()}")
    logger.log(f"Fold {args.fold}: train={len(split['train'])} {args.eval_on}={len(split[args.eval_on])}")

    train_loader = make_loader(raw, split["train"], args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = make_loader(raw, split[args.eval_on], args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = _make_model(args, spec)
    logger.log(model)

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
        weight_decay=args.weight_decay,
    )
    _annotate_checkpoints(ckpt_dir, {"encoder_cfg": raw.encoder_cfg(), "model_cfg": _model_cfg(args, spec)})

    y_true, y_pred = predict_loader(model, eval_loader, device)
    metrics = evaluate_all(y_true, y_pred, spec["aupr_threshold"])
    logger.log(metrics)
    save_json(
        {
            "args": vars(args),
            "encoder_cfg": raw.encoder_cfg(),
            "metrics": metrics,
            "history": result["history"],
        },
        log_dir / "metrics.json",
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    raw, spec = _load_dataset(args)
    split = paper_splits(raw)[args.fold]
    loader = make_loader(raw, split[args.split], args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    _warn_encoder_mismatch(ckpt, raw.encoder_cfg())
    model = _restore_model(args, spec, ckpt).to(device)

    y_true, y_pred = predict_loader(model, loader, device)
    metrics = evaluate_all(y_true, y_pred, spec["aupr_threshold"])
    print(metrics)
    out = REPO_ROOT / "logs" / "deepdta_pretrain" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "predictions.npz", y_true=y_true, y_pred=y_pred)
    save_json(metrics, out / "metrics.json")
    print(f"Saved predictions to {out / 'predictions.npz'}")


def cmd_predict(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    ckpt = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    saved = ckpt.get("encoder_cfg") or {}
    cfg = ckpt.get("model_cfg") or {}

    def _pick(kind: str, key: str, override: Any, fallback: Any) -> Any:
        if override is not None:
            return override
        return (saved.get(kind) or {}).get(key, fallback)

    drug_spec = encoder_defaults("drug")
    prot_spec = encoder_defaults("protein")
    drug_model = _pick("drug", "hf_name", args.drug_model, drug_spec["hf_name"])
    prot_model = _pick("protein", "hf_name", args.protein_model, prot_spec["hf_name"])
    pool = _pick("drug", "pool", args.pool, "mean")
    max_smi_len = _pick("drug", "max_len", args.max_smi_len, drug_spec["max_len"])
    max_prot_len = _pick("protein", "max_len", args.max_prot_len, prot_spec["max_len"])

    drug_emb, _ = encode_texts(
        [args.smiles], hf_name=drug_model, max_len=max_smi_len, pool=pool,
        device=device, batch_size=1, show_progress=False,
    )
    prot_emb, _ = encode_texts(
        [args.protein], hf_name=prot_model, max_len=max_prot_len, pool=pool,
        device=device, batch_size=1, show_progress=False,
    )

    model = DeepDTAPretrain(
        drug_dim=cfg.get("drug_dim", drug_emb.shape[1]),
        prot_dim=cfg.get("prot_dim", prot_emb.shape[1]),
        proj_dim=cfg.get("proj_dim", 256),
        dropout=cfg.get("dropout", 0.1),
    )
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    drug = torch.from_numpy(drug_emb).to(device)
    protein = torch.from_numpy(prot_emb).to(device)
    with torch.no_grad():
        pred = float(model(drug, protein).cpu().numpy().ravel()[0])
    print(f"Predicted affinity: {pred:.6f}")


def cmd_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    raw, spec = _load_dataset(args)
    splits = paper_splits(raw)
    log_dir, ckpt_dir = _run_dirs(args)
    logger = Logger(log_dir)
    logger.log(vars(args))
    logger.log(f"Device: {device}")
    logger.log(f"Encoders: {raw.encoder_cfg()}")
    logger.log(f"Drugs={raw.n_drugs} proteins={raw.n_proteins} labeled={len(raw.label_row_inds)}")

    test_ci, test_mse = [], []
    for split in splits:
        fold = split["fold"]
        logger.log(f"Fold {fold}: train={len(split['train'])} test={len(split['test'])}")
        set_seed(args.seed)
        train_loader = make_loader(raw, split["train"], args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = make_loader(raw, split["test"], args.batch_size, shuffle=False, num_workers=args.num_workers)
        model = _make_model(args, spec)
        fold_ckpt = ckpt_dir / f"fold{fold}"
        train_one_run(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            checkpoint_dir=fold_ckpt,
            logger=logger,
            weight_decay=args.weight_decay,
        )
        _annotate_checkpoints(fold_ckpt, {"encoder_cfg": raw.encoder_cfg(), "model_cfg": _model_cfg(args, spec)})
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
        {
            "encoder_cfg": raw.encoder_cfg(),
            "test_ci": test_ci,
            "test_mse": test_mse,
            "avg_ci": avg_ci,
            "avg_mse": avg_mse,
            "std_ci": std_ci,
        },
        log_dir / "final.json",
    )


def main() -> None:
    args = build_parser().parse_args()
    {
        "precompute": cmd_precompute,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "experiment": cmd_experiment,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
