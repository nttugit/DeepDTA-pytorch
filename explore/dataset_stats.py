#!/usr/bin/env python3
"""Summarize Davis and KIBA folders under data/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.dataset import (
    DRUG_SIM_CANDIDATES,
    PROTEIN_SIM_CANDIDATES,
    load_affinity,
    load_folds,
    load_json_ordered,
    load_similarity_matrix,
)
from preprocessing.encoding import log_transform_kd
from utils import ensure_dir, save_json

PRESETS: dict[str, dict[str, Any]] = {
    "davis": {
        "subdir": "davis",
        "is_log": True,
        "raw_name": "Kd (nM)",
        "train_name": "pKd",
        "aupr_threshold": 7.0,
        "max_smi_len": 85,
        "max_seq_len": 1200,
        "assay_ceiling": 10000.0,
    },
    "kiba": {
        "subdir": "kiba",
        "is_log": False,
        "raw_name": "KIBA score",
        "train_name": "KIBA score",
        "aupr_threshold": 12.1,
        "max_smi_len": 100,
        "max_seq_len": 1000,
        "assay_ceiling": None,
    },
}


def _length_stats(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.int64)
    return {
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": int(arr.max()),
    }


def _percentiles(arr: np.ndarray) -> dict[str, float]:
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    values = np.percentile(arr, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, values)}


def _count_stats(counts: np.ndarray) -> dict[str, float]:
    return {
        "min": int(counts.min()) if counts.size else 0,
        "median": float(np.median(counts)) if counts.size else 0.0,
        "mean": float(counts.mean()) if counts.size else 0.0,
        "max": int(counts.max()) if counts.size else 0,
        "zeros": int((counts == 0).sum()),
    }


def summarize_dataset(name: str, data_root: Path) -> dict[str, Any]:
    preset = PRESETS[name]
    folder = data_root / preset["subdir"]
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")

    ligand_path = folder / "ligands_can.txt"
    if not ligand_path.exists():
        ligand_path = folder / "ligands.txt"
    protein_path = folder / "proteins.txt"
    affinity_path = folder / "Y"
    for path in (ligand_path, protein_path, affinity_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

    ligands = load_json_ordered(ligand_path)
    proteins = load_json_ordered(protein_path)
    y_raw = load_affinity(affinity_path)
    labeled_mask = np.isfinite(y_raw)
    labeled = y_raw[labeled_mask]
    n_drugs = len(ligands)
    n_proteins = len(proteins)
    n_cells = int(y_raw.size)
    n_labeled = int(labeled_mask.sum())
    n_nan = int(np.isnan(y_raw).sum())

    smiles = list(ligands.values())
    sequences = list(proteins.values())
    smiles_lens = [len(s) for s in smiles]
    seq_lens = [len(s) for s in sequences]
    protein_ids = list(proteins.keys())
    mutant_like = sum("(" in pid for pid in protein_ids)
    gene_like = len({pid.split("(")[0] for pid in protein_ids})

    y_train = log_transform_kd(y_raw) if preset["is_log"] else y_raw
    train_vals = y_train[np.isfinite(y_train)]
    threshold = float(preset["aupr_threshold"])
    n_pos = int((train_vals >= threshold).sum())

    pairs_per_drug = labeled_mask.sum(axis=1)
    pairs_per_protein = labeled_mask.sum(axis=0)

    fold_info: dict[str, Any] = {"available": False}
    try:
        test_fold, train_folds = load_folds(folder, problem_type=1)
        fold_info = {
            "available": True,
            "test_size": len(test_fold),
            "n_train_folds": len(train_folds),
            "train_fold_sizes": [len(fold) for fold in train_folds],
            "paper_train_size": int(sum(len(f) for f in train_folds[1:])) if len(train_folds) > 1 else None,
            "paper_val_size": len(train_folds[0]) if train_folds else None,
        }
    except FileNotFoundError:
        pass

    drug_sim = load_similarity_matrix(folder, DRUG_SIM_CANDIDATES)
    protein_sim = load_similarity_matrix(folder, PROTEIN_SIM_CANDIDATES)
    ceiling = preset["assay_ceiling"]
    ceiling_frac = float((labeled == ceiling).mean()) if ceiling is not None and labeled.size else None

    stats: dict[str, Any] = {
        "name": name,
        "path": str(folder),
        "n_drugs": n_drugs,
        "n_proteins": n_proteins,
        "y_shape": list(y_raw.shape),
        "n_matrix_cells": n_cells,
        "n_labeled_pairs": n_labeled,
        "n_nan": n_nan,
        "fill_rate": float(n_labeled / n_cells) if n_cells else 0.0,
        "raw_affinity": {
            "name": preset["raw_name"],
            "min": float(labeled.min()) if labeled.size else None,
            "max": float(labeled.max()) if labeled.size else None,
            "median": float(np.median(labeled)) if labeled.size else None,
            "mean": float(labeled.mean()) if labeled.size else None,
            "n_unique": int(len(np.unique(labeled))) if labeled.size else 0,
            "percentiles": _percentiles(labeled) if labeled.size else {},
            "assay_ceiling": ceiling,
            "assay_ceiling_fraction": ceiling_frac,
        },
        "train_affinity": {
            "name": preset["train_name"],
            "is_log": bool(preset["is_log"]),
            "min": float(train_vals.min()) if train_vals.size else None,
            "max": float(train_vals.max()) if train_vals.size else None,
            "median": float(np.median(train_vals)) if train_vals.size else None,
            "mean": float(train_vals.mean()) if train_vals.size else None,
            "percentiles": _percentiles(train_vals) if train_vals.size else {},
            "aupr_threshold": threshold,
            "n_positive": n_pos,
            "positive_rate": float(n_pos / train_vals.size) if train_vals.size else 0.0,
        },
        "smiles_length": _length_stats(smiles_lens),
        "protein_length": _length_stats(seq_lens),
        "truncation": {
            "max_smi_len": preset["max_smi_len"],
            "max_seq_len": preset["max_seq_len"],
            "n_smiles_truncated": int(sum(n > preset["max_smi_len"] for n in smiles_lens)),
            "n_proteins_truncated": int(sum(n > preset["max_seq_len"] for n in seq_lens)),
        },
        "pairs_per_drug": _count_stats(pairs_per_drug),
        "pairs_per_protein": _count_stats(pairs_per_protein),
        "ids": {
            "example_drug_ids": list(ligands.keys())[:8],
            "example_protein_ids": protein_ids[:8],
            "example_smiles": smiles[0][:80] if smiles else "",
            "n_mutant_like_proteins": mutant_like,
            "n_gene_like_proteins": gene_like,
        },
        "folds": fold_info,
        "similarity": {
            "drug_sim_shape": list(drug_sim.shape) if drug_sim is not None else None,
            "protein_sim_shape": list(protein_sim.shape) if protein_sim is not None else None,
        },
        "_arrays": {
            "train_vals": train_vals,
            "smiles_lens": np.asarray(smiles_lens, dtype=np.int64),
            "seq_lens": np.asarray(seq_lens, dtype=np.int64),
        },
    }
    return stats


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value >= 1000 or value <= -1000:
            return f"{value:,.{digits}f}"
        return f"{value:.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def print_dataset(stats: dict[str, Any]) -> None:
    raw = stats["raw_affinity"]
    train = stats["train_affinity"]
    folds = stats["folds"]
    trunc = stats["truncation"]
    print("=" * 72)
    print(f"{stats['name'].upper()}  ({stats['path']})")
    print("=" * 72)
    print(f"  drugs / proteins / Y shape     {stats['n_drugs']:,} / {stats['n_proteins']:,} / {tuple(stats['y_shape'])}")
    print(f"  labeled pairs / NaN / fill     {_fmt(stats['n_labeled_pairs'])} / {_fmt(stats['n_nan'])} / {stats['fill_rate']:.1%}")
    print()
    print(f"  raw affinity ({raw['name']})")
    print(f"    min / median / mean / max    {_fmt(raw['min'])} / {_fmt(raw['median'])} / {_fmt(raw['mean'])} / {_fmt(raw['max'])}")
    print(f"    unique values                {_fmt(raw['n_unique'])}")
    if raw["assay_ceiling"] is not None:
        print(
            f"    assay ceiling {raw['assay_ceiling']:g}         "
            f"{raw['assay_ceiling_fraction']:.1%} of labeled pairs"
        )
    print()
    log_note = "Kd → pKd = -log10(Kd / 1e9)" if train["is_log"] else "already transformed; is_log=0"
    print(f"  training labels ({train['name']}; {log_note})")
    print(f"    min / median / mean / max    {_fmt(train['min'])} / {_fmt(train['median'])} / {_fmt(train['mean'])} / {_fmt(train['max'])}")
    print(
        f"    AUPR positives ≥ {train['aupr_threshold']}       "
        f"{_fmt(train['n_positive'])} ({train['positive_rate']:.1%})"
    )
    pct = train["percentiles"]
    if pct:
        keys = ["p0", "p5", "p25", "p50", "p75", "p95", "p100"]
        print("    percentiles                  " + "  ".join(f"{k}={_fmt(pct[k], 3)}" for k in keys))
    print()
    smi = stats["smiles_length"]
    seq = stats["protein_length"]
    print("  SMILES length                   min/median/max  "
          f"{smi['min']} / {smi['median']:.0f} / {smi['max']}")
    print("  protein length                  min/median/max  "
          f"{seq['min']} / {seq['median']:.0f} / {seq['max']}")
    print(
        f"  truncated vs config             SMILES > {trunc['max_smi_len']}: "
        f"{trunc['n_smiles_truncated']:,}   proteins > {trunc['max_seq_len']}: "
        f"{trunc['n_proteins_truncated']:,}"
    )
    print()
    dpd = stats["pairs_per_drug"]
    ppp = stats["pairs_per_protein"]
    print("  labeled pairs / drug            min/median/max  "
          f"{dpd['min']} / {dpd['median']:.0f} / {dpd['max']}   (empty={dpd['zeros']})")
    print("  labeled pairs / protein         min/median/max  "
          f"{ppp['min']} / {ppp['median']:.0f} / {ppp['max']}   (empty={ppp['zeros']})")
    ids = stats["ids"]
    print(f"  example drug IDs               {', '.join(ids['example_drug_ids'][:5])}")
    print(f"  example protein IDs            {', '.join(ids['example_protein_ids'][:5])}")
    print(
        f"  protein ID variants             gene-like={ids['n_gene_like_proteins']:,}  "
        f"mutant-like={ids['n_mutant_like_proteins']:,}"
    )
    print()
    if folds.get("available"):
        print(f"  test fold                      {folds['test_size']:,} pairs (fixed)")
        print(f"  train folds                    {folds['n_train_folds']} × {folds['train_fold_sizes']}")
        if folds.get("paper_train_size"):
            print(
                f"  paper split (fold 0)           train={folds['paper_train_size']:,}  "
                f"val={folds['paper_val_size']:,}  test={folds['test_size']:,}"
            )
    else:
        print("  folds                           not found")
    sim = stats["similarity"]
    print(f"  similarity matrices             drug={sim['drug_sim_shape']}  protein={sim['protein_sim_shape']}")
    print()


def print_comparison(all_stats: list[dict[str, Any]]) -> None:
    if len(all_stats) < 2:
        return
    headers = ["", *[s["name"].upper() for s in all_stats]]
    rows = [
        ("drugs", [s["n_drugs"] for s in all_stats]),
        ("proteins", [s["n_proteins"] for s in all_stats]),
        ("labeled pairs", [s["n_labeled_pairs"] for s in all_stats]),
        ("NaN cells", [s["n_nan"] for s in all_stats]),
        ("fill rate", [f"{s['fill_rate']:.1%}" for s in all_stats]),
        ("raw affinity", [s["raw_affinity"]["name"] for s in all_stats]),
        ("train label", [s["train_affinity"]["name"] for s in all_stats]),
        ("train min", [s["train_affinity"]["min"] for s in all_stats]),
        ("train median", [s["train_affinity"]["median"] for s in all_stats]),
        ("train max", [s["train_affinity"]["max"] for s in all_stats]),
        ("AUPR threshold", [s["train_affinity"]["aupr_threshold"] for s in all_stats]),
        ("AUPR positives", [f"{s['train_affinity']['n_positive']:,} ({s['train_affinity']['positive_rate']:.1%})" for s in all_stats]),
        ("test pairs", [s["folds"].get("test_size", "-") for s in all_stats]),
        ("SMILES max (data/cfg)", [f"{s['smiles_length']['max']}/{s['truncation']['max_smi_len']}" for s in all_stats]),
        ("protein max (data/cfg)", [f"{s['protein_length']['max']}/{s['truncation']['max_seq_len']}" for s in all_stats]),
    ]
    col0 = max(len(h) for h, _ in rows)
    colw = 28
    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(f"{headers[0]:<{col0}}  " + "  ".join(f"{h:>{colw}}" for h in headers[1:]))
    for label, values in rows:
        cells = [_fmt(v) if not isinstance(v, str) else v for v in values]
        print(f"{label:<{col0}}  " + "  ".join(f"{c:>{colw}}" for c in cells))
    print()


def save_plots(all_stats: list[dict[str, Any]], out_dir: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skip plots")
        return None

    ensure_dir(out_dir)
    n = len(all_stats)
    fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 10.5), squeeze=False)
    for col, stats in enumerate(all_stats):
        arrays = stats["_arrays"]
        name = stats["name"].upper()
        axes[0, col].hist(arrays["train_vals"], bins=40, color="#3b6ea5", edgecolor="white")
        axes[0, col].axvline(stats["train_affinity"]["aupr_threshold"], color="#c0392b", linestyle="--", label="AUPR threshold")
        axes[0, col].set_title(f"{name}  {stats['train_affinity']['name']}")
        axes[0, col].set_xlabel(stats["train_affinity"]["name"])
        axes[0, col].set_ylabel("pairs")
        axes[0, col].legend(frameon=False)

        axes[1, col].hist(arrays["smiles_lens"], bins=30, color="#2e8b57", edgecolor="white")
        axes[1, col].axvline(stats["truncation"]["max_smi_len"], color="#c0392b", linestyle="--", label="max_smi_len")
        axes[1, col].set_title(f"{name}  SMILES length")
        axes[1, col].set_xlabel("characters")
        axes[1, col].set_ylabel("drugs")
        axes[1, col].legend(frameon=False)

        axes[2, col].hist(arrays["seq_lens"], bins=30, color="#8e5a2b", edgecolor="white")
        axes[2, col].axvline(stats["truncation"]["max_seq_len"], color="#c0392b", linestyle="--", label="max_seq_len")
        axes[2, col].set_title(f"{name}  protein length")
        axes[2, col].set_xlabel("amino acids")
        axes[2, col].set_ylabel("proteins")
        axes[2, col].legend(frameon=False)

    fig.tight_layout()
    path = out_dir / "davis_kiba_stats.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _jsonable(stats: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in stats.items() if k != "_arrays"}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Davis / KIBA dataset statistics.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Folder that contains davis/ and kiba/ (default: data)",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["davis", "kiba"],
        default=["davis", "kiba"],
        help="Which datasets to summarize",
    )
    parser.add_argument("--save-plots", action="store_true", help="Write histograms to explore/figures/")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write stats JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    all_stats = [summarize_dataset(name, data_root) for name in args.dataset]
    print_comparison(all_stats)
    for stats in all_stats:
        print_dataset(stats)

    if args.save_plots:
        plot_path = save_plots(all_stats, ROOT / "explore" / "figures")
        if plot_path is not None:
            print(f"saved plots: {plot_path}")

    if args.json_out:
        payload = [_jsonable(s) for s in all_stats]
        out = Path(args.json_out)
        if not out.is_absolute():
            out = ROOT / out
        save_json(payload, out)
        print(f"saved json: {out}")


if __name__ == "__main__":
    main()
