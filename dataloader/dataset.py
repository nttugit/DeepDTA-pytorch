"""Davis / KIBA / custom DeepDTA datasets and PyTorch DataLoaders."""

from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing.charset import CHARISOSMILEN, CHARPROTLEN
from preprocessing.encoding import encode_all, log_transform_kd

DRUG_SIM_CANDIDATES = (
    "drug-drug_similarities_2D.txt",
    "kiba_drug_sim.txt",
)
PROTEIN_SIM_CANDIDATES = (
    "target-target_similarities_WS.txt",
    "kiba_target_sim.txt",
)


def _as_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    return path if path.is_dir() else path.parent


def load_json_ordered(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=OrderedDict)


def load_affinity(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        try:
            y = pickle.load(handle, encoding="latin1")
        except TypeError:
            y = pickle.load(handle)
    return np.asarray(y, dtype=np.float64)


def load_similarity_matrix(dataset_dir: Path, candidates: Sequence[str]) -> Optional[np.ndarray]:
    for name in candidates:
        path = dataset_dir / name
        if path.exists():
            return np.loadtxt(path, dtype=np.float32)
    return None


def load_folds(dataset_dir: Path, problem_type: int = 1) -> tuple[list[int], list[list[int]]]:
    """Load paper splits. Returned indices index *labeled pairs*, not drugs/proteins.

    ``problem_type=1`` is the paper's default setting (random pair split).
    """
    folds_dir = dataset_dir / "folds"
    setting_test = folds_dir / f"test_fold_setting{problem_type}.txt"
    setting_train = folds_dir / f"train_fold_setting{problem_type}.txt"
    plain_test = folds_dir / "test_fold.txt"
    plain_train = folds_dir / "train_fold.txt"

    if setting_test.exists():
        test_fold = json.loads(setting_test.read_text())
    elif plain_test.exists():
        test_fold = json.loads(plain_test.read_text())
    else:
        raise FileNotFoundError(f"No test fold file found under {folds_dir}")

    if setting_train.exists():
        train_folds = json.loads(setting_train.read_text())
    elif plain_train.exists():
        raw = json.loads(plain_train.read_text())
        train_folds = raw if raw and isinstance(raw[0], list) else [raw]
    else:
        raise FileNotFoundError(f"No train fold file found under {folds_dir}")

    return list(test_fold), [list(fold) for fold in train_folds]


class DeepDTARawData:
    """In-memory DeepDTA dataset: unique sequences encoded once, pairs looked up later.

    Parameters
    ----------
    max_smi_len, max_seq_len
        Pad / truncate lengths (KIBA 100/1000, Davis 85/1200).
    is_log
        1 = Davis ``pKd = -log10(Kd / 1e9)``. 0 = KIBA ``Y`` already transformed.
    encoding
        ``label`` → integer ids for Embedding; ``onehot`` → (L, vocab) matrices.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        max_smi_len: int = 100,
        max_seq_len: int = 1000,
        is_log: int = 0,
        problem_type: int = 1,
        ligands_file: str = "ligands_can.txt",
        proteins_file: str = "proteins.txt",
        affinity_file: str = "Y",
        encoding: str = "label",
    ):
        self.dataset_dir = _as_dir(dataset_path)
        self.max_smi_len = max_smi_len
        self.max_seq_len = max_seq_len
        self.problem_type = problem_type
        self.encoding = encoding
        self.charseqset_size = CHARPROTLEN
        self.charsmiset_size = CHARISOSMILEN

        ligand_path = self.dataset_dir / ligands_file
        if not ligand_path.exists():
            alt = self.dataset_dir / "ligands.txt"
            if alt.exists():
                ligand_path = alt
        protein_path = self.dataset_dir / proteins_file
        affinity_path = self.dataset_dir / affinity_file

        ligands = load_json_ordered(ligand_path)
        proteins = load_json_ordered(protein_path)
        self.drug_ids = list(ligands.keys())
        self.protein_ids = list(proteins.keys())
        self.smiles = [ligands[k] for k in self.drug_ids]
        self.sequences = [proteins[k] for k in self.protein_ids]

        y = load_affinity(affinity_path)  # pickle matrix, drugs × proteins
        if int(is_log):
            y = log_transform_kd(y)
        self.Y = np.asarray(y, dtype=np.float64)

        with_label = encoding != "onehot"
        # XD/XT are unique-entity tables; pair samples index into these rows.
        self.XD, self.XT = encode_all(
            self.smiles,
            self.sequences,
            max_smi_len=max_smi_len,
            max_seq_len=max_seq_len,
            with_label=with_label,
        )
        self.drug_sim = load_similarity_matrix(self.dataset_dir, DRUG_SIM_CANDIDATES)
        self.protein_sim = load_similarity_matrix(self.dataset_dir, PROTEIN_SIM_CANDIDATES)

        # Flatten labeled (non-NaN) cells. Fold files index this list, not Y itself.
        rows, cols = np.where(np.isnan(self.Y) == False)  # noqa: E712
        self.label_row_inds = np.asarray(rows, dtype=np.int64)
        self.label_col_inds = np.asarray(cols, dtype=np.int64)

    @property
    def n_drugs(self) -> int:
        return int(self.XD.shape[0])

    @property
    def n_proteins(self) -> int:
        return int(self.XT.shape[0])

    def read_sets(self) -> tuple[list[int], list[list[int]]]:
        return load_folds(self.dataset_dir, self.problem_type)

    def pair_arrays(self, pair_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map fold pair-ids → (drug_row, protein_col, affinity)."""
        pair_indices = np.asarray(pair_indices, dtype=np.int64)
        rows = self.label_row_inds[pair_indices]
        cols = self.label_col_inds[pair_indices]
        affinities = self.Y[rows, cols].astype(np.float32)
        return rows, cols, affinities


class DTAPairDataset(Dataset):
    """One drug–protein pair per item. ``drug``/``protein`` are already encoded tensors."""

    def __init__(self, raw: DeepDTARawData, pair_indices: Sequence[int]):
        self.raw = raw
        self.drug_idx, self.protein_idx, self.affinity = raw.pair_arrays(pair_indices)

    def __len__(self) -> int:
        return int(self.affinity.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        d_idx = int(self.drug_idx[index])
        p_idx = int(self.protein_idx[index])
        sample = {
            "drug": torch.as_tensor(self.raw.XD[d_idx]),  # (L_smi,) or (L_smi, vocab)
            "protein": torch.as_tensor(self.raw.XT[p_idx]),
            "affinity": torch.tensor(self.affinity[index], dtype=torch.float32),
            "drug_idx": torch.tensor(d_idx, dtype=torch.long),  # ablation / similarity models
            "protein_idx": torch.tensor(p_idx, dtype=torch.long),
        }
        if self.raw.drug_sim is not None:
            sample["drug_sim"] = torch.as_tensor(self.raw.drug_sim[d_idx], dtype=torch.float32)
        if self.raw.protein_sim is not None:
            sample["protein_sim"] = torch.as_tensor(self.raw.protein_sim[p_idx], dtype=torch.float32)
        return sample


def make_dataloader(
    raw: DeepDTARawData,
    pair_indices: Sequence[int],
    batch_size: int = 256,
    shuffle: bool = False,  # paper default: do not shuffle batches
    num_workers: int = 0,
) -> DataLoader:
    dataset = DTAPairDataset(raw, pair_indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def paper_splits(raw: DeepDTARawData) -> list[dict[str, Any]]:
    """Paper protocol: 5 train folds rotated as val; the same held-out test every time.

    Each returned dict has pair-index lists ``train`` / ``val`` / ``test``.
    """
    test_fold, train_folds = raw.read_sets()
    if len(train_folds) == 1:
        ids = list(train_folds[0])
        cut = max(1, int(0.8 * len(ids)))
        if cut >= len(ids):
            cut = max(1, len(ids) - 1)
        return [
            {
                "fold": 0,
                "train": ids[:cut],
                "val": ids[cut:] or ids[-1:],
                "test": list(test_fold),
            }
        ]
    splits = []
    for val_idx, val_fold in enumerate(train_folds):
        train_ids = [i for j, fold in enumerate(train_folds) if j != val_idx for i in fold]
        splits.append(
            {
                "fold": val_idx,
                "train": train_ids,
                "val": list(val_fold),
                "test": list(test_fold),
            }
        )
    return splits
