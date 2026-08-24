"""Shared helpers for configuration, logging, seeding, and checkpoints."""

from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "" or text in {"null", "None", "~"}:
        return None
    if text.lower() in {"true", "yes", "on"}:
        return True
    if text.lower() in {"false", "no", "off"}:
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_simple_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Parse the project's indented YAML configs without PyYAML.

    Training on Cheaha only needs torch + numpy. This covers scalars, inline
    lists, and nested maps used in ``configs/*.yaml``.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if ":" not in stripped:
            raise ValueError(f"Cannot parse YAML line in {path}: {raw}")
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        # Allow ``key: value  # comment`` (full-line ``#`` comments are already skipped).
        if rest and not (rest.startswith('"') or rest.startswith("'")) and " #" in rest:
            rest = rest.split(" #", 1)[0].rstrip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if rest == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root

# Defaults merged under every YAML. Dataset-specific files only override what differs.
# is_log: Davis=1 (Kd→pKd), KIBA=0. shuffle/restore_best_weights=False match the Keras protocol.
DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1,
    "device": "auto",
    "dataset": {
        "name": "kiba",
        "path": "data/kiba/",
        "problem_type": 1,
        "is_log": 0,
        "max_smi_len": 100,
        "max_seq_len": 1000,
        "ligands_file": "ligands_can.txt",
        "proteins_file": "proteins.txt",
        "affinity_file": "Y",
        "encoding": "label",
        "aupr_threshold": 12.1,
    },
    "model": {
        "name": "combined_categorical",
        "embedding_dim": 128,
        "num_filters": 32,
        "smi_filter_length": 8,
        "seq_filter_length": 12,
        "dropout": 0.1,
        "fc_dims": [1024, 1024, 512],
        "keras_init": True,
    },
    "train": {
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "patience": 15,
        "shuffle": False,
        "num_workers": 0,
        "restore_best_weights": False,
        "log_interval": 1,
    },
    "search": {
        "num_windows": [32],
        "smi_window_lengths": [4, 8],
        "seq_window_lengths": [8, 12],
    },
    "paths": {
        "log_dir": "logs",
        "checkpoint_dir": "checkpoints",
        "figure_dir": "logs/figures",
    },
}


def deep_update(base: MutableMapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(base))
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """YAML over ``DEFAULT_CONFIG``. Missing keys keep the defaults above."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        cfg_path = Path(path)
        loaded = load_simple_yaml(cfg_path) or {}
        cfg = deep_update(cfg, loaded)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def resolve_path(path: Union[str, Path], root: Path = ROOT) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    return path


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 1) -> None:
    """Match the original DeepDTA seeding as closely as PyTorch allows."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    """Pick a device that this Torch build actually supports.

    CPU-only wheels raise ``AssertionError: Torch not compiled with CUDA enabled``
    if we blindly call ``.to('cuda')``. Requesting ``cuda`` therefore falls back
    to CPU with a warning instead of crashing the Slurm job.
    """
    requested = (name or "auto").lower()
    cuda_ok = torch.cuda.is_available()
    if requested in {"auto", "cuda"} and cuda_ok:
        return torch.device("cuda")
    if requested == "cuda" and not cuda_ok:
        print(
            "WARNING: --device cuda was requested but this PyTorch build has no CUDA.\n"
            "Falling back to CPU. On Cheaha install a GPU wheel:\n"
            "  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121"
        )
    if requested == "mps" or (
        requested == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        return torch.device(name)
    return torch.device("cpu")


class Logger:
    def __init__(self, log_dir: Union[str, Path], filename: str = "log.txt"):
        self.log_dir = ensure_dir(log_dir)
        self.path = self.log_dir / filename

    def log(self, msg: Any) -> None:
        text = str(msg)
        print(text)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def save_checkpoint(state: Mapping[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(dict(state), path)


def load_checkpoint(path: Union[str, Path], map_location: Optional[str] = None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
