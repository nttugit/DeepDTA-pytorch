"""Paper protocol: hyper-parameter search on 5-fold CV, then evaluate on the held-out test set.

This is a PyTorch port of ``nfold_1_2_3_setting_sample`` / ``general_nfold_cv``
from the original DeepDTA ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.dataset import DeepDTARawData, make_dataloader, paper_splits
from evaluation.metrics import evaluate_all
from models.deepdta import build_model
from run.trainer import predict_loader, train_one_run
from utils import Logger, get_device, load_config, resolve_path, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepDTA n-fold experiment (PyTorch)")
    parser.add_argument("--config", type=str, default="configs/kiba.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset-path", "--dataset_path", type=str, default=None)
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip val grid search; train/test with YAML num_filters and kernels",
    )
    parser.add_argument(
        "--num-windows",
        "--num_windows",
        type=int,
        nargs="+",
        default=None,
        help="Grid of first-layer filter counts, e.g. --num-windows 32",
    )
    parser.add_argument("--smi-window-lengths", "--smi_window_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--seq-window-lengths", "--seq_window_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--num-epoch", "--num_epoch", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None)
    parser.add_argument("--max-seq-len", "--max_seq_len", type=int, default=None)
    parser.add_argument("--max-smi-len", "--max_smi_len", type=int, default=None)
    parser.add_argument("--problem-type", "--problem_type", type=int, default=None)
    parser.add_argument("--is-log", "--is_log", type=int, default=None)
    parser.add_argument("--log-dir", "--log_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()

def apply_cli(cfg: dict, args: argparse.Namespace) -> dict:
    if args.device:
        cfg["device"] = args.device
    if args.dataset_path:
        cfg["dataset"]["path"] = args.dataset_path
    if args.num_epoch is not None:
        cfg["train"]["epochs"] = args.num_epoch
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_windows is not None:
        cfg["search"]["num_windows"] = args.num_windows
    if args.smi_window_lengths is not None:
        cfg["search"]["smi_window_lengths"] = args.smi_window_lengths
    if args.seq_window_lengths is not None:
        cfg["search"]["seq_window_lengths"] = args.seq_window_lengths
    if args.max_seq_len is not None:
        cfg["dataset"]["max_seq_len"] = args.max_seq_len
    if args.max_smi_len is not None:
        cfg["dataset"]["max_smi_len"] = args.max_smi_len
    if args.problem_type is not None:
        cfg["dataset"]["problem_type"] = args.problem_type
    if args.is_log is not None:
        cfg["dataset"]["is_log"] = args.is_log
    if args.log_dir is not None:
        cfg["paths"]["log_dir"] = args.log_dir
    if args.model is not None:
        cfg["model"]["name"] = args.model
    return cfg


def _make_model(cfg: dict, raw: DeepDTARawData, num_filters: int, smi_k: int, seq_k: int):
    return build_model(
        name=cfg["model"]["name"],
        num_filters=num_filters,
        smi_filter_length=smi_k,
        seq_filter_length=seq_k,
        n_drugs=raw.n_drugs,
        n_proteins=raw.n_proteins,
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
        fc_dims=cfg["model"]["fc_dims"],
        keras_init=cfg["model"]["keras_init"],
        smi_vocab_size=raw.charsmiset_size + 1,
        seq_vocab_size=raw.charseqset_size + 1,
    )


def general_nfold_cv(
    cfg: dict,
    raw: DeepDTARawData,
    splits: list[dict],
    eval_key: str,  # "val" during search, "test" for the final 5 runs
    param_grid: list[tuple[int, int, int]],  # (num_filters, smi_kernel, seq_kernel)
    device,
    logger: Logger,
    fig_dir: Path,
    ckpt_dir: Path,
) -> tuple[int, tuple[int, int, int], list[list[float]], list[list[float]]]:
    """Train every (fold × hyper-param) combo. Best combo = highest *mean* CI across folds."""
    n_params = len(param_grid)
    n_folds = len(splits)
    all_ci = [[0.0 for _ in range(n_folds)] for _ in range(n_params)]
    all_mse = [[0.0 for _ in range(n_folds)] for _ in range(n_params)]

    for foldind, split in enumerate(splits):
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
        logger.log(f"Fold {foldind}: train={len(split['train'])} {eval_key}={len(split[eval_key])}")

        for pointer, (num_filters, smi_k, seq_k) in enumerate(param_grid):
            set_seed(int(cfg["seed"]))
            model = _make_model(cfg, raw, num_filters, smi_k, seq_k)
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
                aupr_threshold=cfg["dataset"]["aupr_threshold"],
                checkpoint_path=ckpt_dir / f"{eval_key}_fold{foldind}_p{pointer}.pt",
                figure_path=fig_dir / f"{eval_key}_b{num_filters}_e{smi_k}_{seq_k}_{foldind}.png",
                logger=logger,
            )
            y_true, y_pred = predict_loader(model, eval_loader, device, raw.n_drugs, raw.n_proteins)
            metrics = evaluate_all(y_true, y_pred, cfg["dataset"]["aupr_threshold"])
            logger.log(
                "P_filters=%d smi_k=%d seq_k=%d Fold=%d CI=%.5f MSE=%.5f rm2=%.5f AUPR=%.5f"
                % (
                    num_filters,
                    smi_k,
                    seq_k,
                    foldind,
                    metrics["ci"],
                    metrics["mse"],
                    metrics["rm2"],
                    metrics["aupr"],
                )
            )
            all_ci[pointer][foldind] = metrics["ci"]
            all_mse[pointer][foldind] = metrics["mse"]
            del model

    best_ci = -float("inf")
    best_pointer = 0
    for pointer, params in enumerate(param_grid):
        avg = float(np.mean(all_ci[pointer]))
        if avg > best_ci:
            best_ci = avg
            best_pointer = pointer
    return best_pointer, param_grid[best_pointer], all_ci, all_mse


def main() -> None:
    args = parse_args()
    cfg = apply_cli(load_config(resolve_path(args.config)), args)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])

    stamp = str(time.time())
    log_dir = resolve_path(cfg["paths"]["log_dir"]) / stamp
    ckpt_dir = resolve_path(cfg["paths"]["checkpoint_dir"]) / stamp
    fig_dir = resolve_path(cfg["paths"]["figure_dir"]) / stamp
    logger = Logger(log_dir)
    logger.log(cfg)
    logger.log(f"Device: {device}")

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
    logger.log(f"Drugs={raw.n_drugs} proteins={raw.n_proteins} labeled={len(raw.label_row_inds)}")
    splits = paper_splits(raw)

    if args.skip_search:
        # Use YAML kernels/filters as-is (quick.yaml / one-fold debugging).
        param_grid = [
            (
                int(cfg["model"]["num_filters"]),
                int(cfg["model"]["smi_filter_length"]),
                int(cfg["model"]["seq_filter_length"]),
            )
        ]
        best_pointer, best_params = 0, param_grid[0]
        logger.log(f"Skipping search; using {best_params}")
    else:
        param_grid = [
            (n, s, p)
            for n in cfg["search"]["num_windows"]
            for s in cfg["search"]["smi_window_lengths"]
            for p in cfg["search"]["seq_window_lengths"]
        ]
        logger.log("---Parameter Search-----")
        logger.log(f"Grid size={len(param_grid)}: {param_grid}")
        best_pointer, best_params, val_ci, val_mse = general_nfold_cv(
            cfg, raw, splits, "val", param_grid, device, logger, fig_dir, ckpt_dir
        )
        logger.log(f"Best val params index={best_pointer} params={best_params}")
        save_json({"val_ci": val_ci, "val_mse": val_mse, "param_grid": param_grid}, log_dir / "search.json")

    logger.log("---Test evaluation with best hyper-parameters-----")
    # Retrain 5 times with the chosen kernels, evaluating the *same* held-out test fold.
    test_grid = [best_params]
    best_pointer, best_params, test_ci, test_mse = general_nfold_cv(
        cfg, raw, splits, "test", test_grid, device, logger, fig_dir, ckpt_dir
    )
    avg_ci = float(np.mean(test_ci[0]))
    avg_mse = float(np.mean(test_mse[0]))
    std_ci = float(np.std(test_ci[0]))
    logger.log("---FINAL RESULTS-----")
    logger.log(f"best param = {best_params}")
    logger.log(f"Test Performance CI: {test_ci[0]}")
    logger.log(f"Test Performance MSE: {test_mse[0]}")
    logger.log("Setting %s" % ds_cfg["name"])
    logger.log("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" % (avg_ci, avg_mse, std_ci))
    save_json(
        {
            "best_params": {
                "num_filters": best_params[0],
                "smi_filter_length": best_params[1],
                "seq_filter_length": best_params[2],
            },
            "test_ci": test_ci[0],
            "test_mse": test_mse[0],
            "avg_ci": avg_ci,
            "avg_mse": avg_mse,
            "std_ci": std_ci,
        },
        log_dir / "final.json",
    )


if __name__ == "__main__":
    main()
