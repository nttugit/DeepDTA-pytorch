"""Davis / KIBA pairs backed by cached ESM-2 and ChemBERTa embeddings.

Same fold protocol and affinity handling as ``deepdta.data``; only the drug and
protein representations change from integer-encoded characters to frozen PLM
vectors.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deepdta.data import (
    DATASETS as CNN_DATASETS,
    REPO_ROOT,
    _load_affinity,
    _load_folds,
    _load_json,
    paper_splits,
    resolve_data_path,
)
from deepdta_pretrain.embeddings import encoder_defaults, get_or_build

__all__ = [
    "DATASETS",
    "REPO_ROOT",
    "PLMData",
    "PLMPairDataset",
    "load_dataset",
    "make_loader",
    "paper_splits",
]

# The CNN-specific knobs (kernel sizes, max_smi_len, max_seq_len) do not apply here.
DATASETS: dict[str, dict[str, Any]] = {
    name: {
        "path": spec["path"],
        "is_log": spec["is_log"],
        "aupr_threshold": spec["aupr_threshold"],
    }
    for name, spec in CNN_DATASETS.items()
}


class PLMData:
    """Unique drugs/proteins embedded once; fold indices select labeled pairs."""

    def __init__(
        self,
        dataset: str,
        dataset_path: Union[str, Path],
        is_log: int = 0,
        drug_model: Optional[str] = None,
        protein_model: Optional[str] = None,
        drug_pool: str = "mean",
        protein_pool: str = "mean",
        max_smi_len: Optional[int] = None,
        max_prot_len: Optional[int] = None,
        device: Union[torch.device, str, None] = "auto",
        encode_batch_size: int = 8,
        rebuild_cache: bool = False,
    ):
        self.dataset = dataset
        self.dataset_dir = Path(dataset_path)
        if not self.dataset_dir.is_dir():
            self.dataset_dir = self.dataset_dir.parent

        ligand_path = self.dataset_dir / "ligands_can.txt"
        if not ligand_path.exists():
            ligand_path = self.dataset_dir / "ligands.txt"
        ligands = _load_json(ligand_path)
        proteins = _load_json(self.dataset_dir / "proteins.txt")
        self.drug_keys = list(ligands.keys())
        self.protein_keys = list(proteins.keys())
        self.smiles = list(ligands.values())
        self.sequences = list(proteins.values())

        y = _load_affinity(self.dataset_dir / "Y")
        if int(is_log):
            y = -np.log10(y / math.pow(10, 9))
        self.Y = np.asarray(y, dtype=np.float64)

        self.drug_emb, self.drug_meta = get_or_build(
            dataset=dataset,
            kind="drug",
            keys=self.drug_keys,
            texts=self.smiles,
            hf_name=drug_model,
            max_len=max_smi_len,
            pool=drug_pool,
            device=device,
            batch_size=encode_batch_size,
            rebuild=rebuild_cache,
        )
        self.protein_emb, self.protein_meta = get_or_build(
            dataset=dataset,
            kind="protein",
            keys=self.protein_keys,
            texts=self.sequences,
            hf_name=protein_model,
            max_len=max_prot_len,
            pool=protein_pool,
            device=device,
            batch_size=encode_batch_size,
            rebuild=rebuild_cache,
        )

        rows, cols = np.where(~np.isnan(self.Y))
        self.label_row_inds = rows.astype(np.int64)
        self.label_col_inds = cols.astype(np.int64)
        self.test_fold, self.train_folds = _load_folds(self.dataset_dir)

    @property
    def n_drugs(self) -> int:
        return int(self.drug_emb.shape[0])

    @property
    def n_proteins(self) -> int:
        return int(self.protein_emb.shape[0])

    @property
    def drug_dim(self) -> int:
        return int(self.drug_emb.shape[1])

    @property
    def protein_dim(self) -> int:
        return int(self.protein_emb.shape[1])

    def encoder_cfg(self) -> dict[str, Any]:
        return {"drug": dict(self.drug_meta), "protein": dict(self.protein_meta)}

    def pair_arrays(self, pair_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pair_indices = np.asarray(pair_indices, dtype=np.int64)
        rows = self.label_row_inds[pair_indices]
        cols = self.label_col_inds[pair_indices]
        return rows, cols, self.Y[rows, cols].astype(np.float32)


class PLMPairDataset(Dataset):
    def __init__(self, raw: PLMData, pair_indices: Sequence[int]):
        self.drug_emb = torch.from_numpy(raw.drug_emb)
        self.protein_emb = torch.from_numpy(raw.protein_emb)
        self.drug_idx, self.protein_idx, self.affinity = raw.pair_arrays(pair_indices)

    def __len__(self) -> int:
        return int(self.affinity.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "drug": self.drug_emb[int(self.drug_idx[index])],
            "protein": self.protein_emb[int(self.protein_idx[index])],
            "affinity": torch.tensor(self.affinity[index], dtype=torch.float32),
        }


def make_loader(
    raw: PLMData,
    pair_indices: Sequence[int],
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        PLMPairDataset(raw, pair_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def load_dataset(
    name: str,
    data_dir: Union[str, Path, None] = None,
    drug_model: Optional[str] = None,
    protein_model: Optional[str] = None,
    drug_pool: str = "mean",
    protein_pool: str = "mean",
    max_smi_len: Optional[int] = None,
    max_prot_len: Optional[int] = None,
    device: Union[torch.device, str, None] = "auto",
    encode_batch_size: int = 8,
    rebuild_cache: bool = False,
) -> tuple[PLMData, dict[str, Any]]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASETS)}")
    spec = dict(DATASETS[name])
    if data_dir is not None:
        spec["path"] = str(data_dir)
    spec["drug_model"] = drug_model or encoder_defaults("drug")["hf_name"]
    spec["protein_model"] = protein_model or encoder_defaults("protein")["hf_name"]
    raw = PLMData(
        dataset=name,
        dataset_path=resolve_data_path(spec["path"]),
        is_log=spec["is_log"],
        drug_model=drug_model,
        protein_model=protein_model,
        drug_pool=drug_pool,
        protein_pool=protein_pool,
        max_smi_len=max_smi_len,
        max_prot_len=max_prot_len,
        device=device,
        encode_batch_size=encode_batch_size,
        rebuild_cache=rebuild_cache,
    )
    spec["drug_dim"] = raw.drug_dim
    spec["protein_dim"] = raw.protein_dim
    return raw, spec
