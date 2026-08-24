"""Load Davis / KIBA pairs and encode SMILES / proteins as integer sequences."""

from __future__ import annotations

import json
import math
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Indices start at 1; 0 is padding. nn.Embedding size is CHARLEN + 1.
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}
CHARPROTLEN = 25

CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64,
}
CHARISOSMILEN = 64

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "kiba": {
        "path": "data/kiba",
        "is_log": 0,
        "max_smi_len": 100,
        "max_seq_len": 1000,
        "smi_kernel": 8,
        "seq_kernel": 12,
        "aupr_threshold": 12.1,
    },
    "davis": {
        "path": "data/davis",
        "is_log": 1,
        "max_smi_len": 85,
        "max_seq_len": 1200,
        "smi_kernel": 4,
        "seq_kernel": 8,
        "aupr_threshold": 7.0,
    },
}


def encode_label(text: str, max_len: int, charset: dict[str, int]) -> np.ndarray:
    ids = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(text[:max_len]):
        if ch in charset:
            ids[i] = charset[ch]
    return ids


def encode_smiles(smiles: str, max_len: int) -> np.ndarray:
    return encode_label(smiles, max_len, CHARISOSMISET)


def encode_protein(sequence: str, max_len: int) -> np.ndarray:
    return encode_label(sequence, max_len, CHARPROTSET)


def _load_json(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=OrderedDict)


def _load_affinity(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        try:
            y = pickle.load(handle, encoding="latin1")
        except TypeError:
            y = pickle.load(handle)
    return np.asarray(y, dtype=np.float64)


def _load_folds(dataset_dir: Path) -> tuple[list[int], list[list[int]]]:
    folds_dir = dataset_dir / "folds"
    test_path = folds_dir / "test_fold_setting1.txt"
    train_path = folds_dir / "train_fold_setting1.txt"
    if not test_path.exists():
        test_path = folds_dir / "test_fold.txt"
    if not train_path.exists():
        train_path = folds_dir / "train_fold.txt"
    test_fold = json.loads(test_path.read_text())
    train_folds = json.loads(train_path.read_text())
    if train_folds and not isinstance(train_folds[0], list):
        train_folds = [train_folds]
    return list(test_fold), [list(fold) for fold in train_folds]


class DTAData:
    """Unique drugs/proteins encoded once; fold indices select labeled pairs."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        max_smi_len: int,
        max_seq_len: int,
        is_log: int = 0,
    ):
        self.dataset_dir = Path(dataset_path)
        if not self.dataset_dir.is_dir():
            self.dataset_dir = self.dataset_dir.parent
        self.max_smi_len = max_smi_len
        self.max_seq_len = max_seq_len

        ligand_path = self.dataset_dir / "ligands_can.txt"
        if not ligand_path.exists():
            ligand_path = self.dataset_dir / "ligands.txt"
        ligands = _load_json(ligand_path)
        proteins = _load_json(self.dataset_dir / "proteins.txt")
        self.smiles = list(ligands.values())
        self.sequences = list(proteins.values())

        y = _load_affinity(self.dataset_dir / "Y")
        if int(is_log):
            y = -np.log10(y / math.pow(10, 9))
        self.Y = np.asarray(y, dtype=np.float64)

        self.XD = np.stack([encode_smiles(s, max_smi_len) for s in self.smiles])
        self.XT = np.stack([encode_protein(p, max_seq_len) for p in self.sequences])

        rows, cols = np.where(~np.isnan(self.Y))
        self.label_row_inds = rows.astype(np.int64)
        self.label_col_inds = cols.astype(np.int64)
        self.test_fold, self.train_folds = _load_folds(self.dataset_dir)

    @property
    def n_drugs(self) -> int:
        return int(self.XD.shape[0])

    @property
    def n_proteins(self) -> int:
        return int(self.XT.shape[0])

    def pair_arrays(self, pair_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pair_indices = np.asarray(pair_indices, dtype=np.int64)
        rows = self.label_row_inds[pair_indices]
        cols = self.label_col_inds[pair_indices]
        return rows, cols, self.Y[rows, cols].astype(np.float32)


class DTAPairDataset(Dataset):
    def __init__(self, raw: DTAData, pair_indices: Sequence[int]):
        self.raw = raw
        self.drug_idx, self.protein_idx, self.affinity = raw.pair_arrays(pair_indices)

    def __len__(self) -> int:
        return int(self.affinity.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        d_idx = int(self.drug_idx[index])
        p_idx = int(self.protein_idx[index])
        return {
            "drug": torch.as_tensor(self.raw.XD[d_idx], dtype=torch.long),
            "protein": torch.as_tensor(self.raw.XT[p_idx], dtype=torch.long),
            "affinity": torch.tensor(self.affinity[index], dtype=torch.float32),
        }


def make_loader(
    raw: DTAData,
    pair_indices: Sequence[int],
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        DTAPairDataset(raw, pair_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def paper_splits(raw: DTAData) -> list[dict[str, Any]]:
    """Rotate each of the 5 train folds as val; the held-out test fold is shared."""
    if len(raw.train_folds) == 1:
        ids = list(raw.train_folds[0])
        cut = max(1, min(int(0.8 * len(ids)), len(ids) - 1))
        return [{"fold": 0, "train": ids[:cut], "val": ids[cut:], "test": list(raw.test_fold)}]

    splits = []
    for val_idx, val_fold in enumerate(raw.train_folds):
        train_ids = [i for j, fold in enumerate(raw.train_folds) if j != val_idx for i in fold]
        splits.append(
            {
                "fold": val_idx,
                "train": train_ids,
                "val": list(val_fold),
                "test": list(raw.test_fold),
            }
        )
    return splits


def resolve_data_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_dataset(name: str, data_dir: Union[str, Path, None] = None) -> tuple[DTAData, dict[str, Any]]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASETS)}")
    spec = dict(DATASETS[name])
    if data_dir is not None:
        spec["path"] = str(data_dir)
    raw = DTAData(
        dataset_path=resolve_data_path(spec["path"]),
        max_smi_len=spec["max_smi_len"],
        max_seq_len=spec["max_seq_len"],
        is_log=spec["is_log"],
    )
    return raw, spec
