"""Sequence encoding helpers ported from the original DeepDTA `datahelper.py`."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from preprocessing.charset import CHARISOSMISET, CHARPROTSET


def one_hot_smiles(line: str, max_smi_len: int, smi_ch_ind: Mapping[str, int]) -> np.ndarray:
    """One-hot SMILES: shape ``(max_smi_len, vocab)``. Column is ``charset[ch] - 1``."""
    x = np.zeros((max_smi_len, len(smi_ch_ind)), dtype=np.float32)
    for i, ch in enumerate(line[:max_smi_len]):  # longer strings are truncated
        if ch in smi_ch_ind:
            x[i, smi_ch_ind[ch] - 1] = 1.0
    return x


def one_hot_sequence(line: str, max_seq_len: int, smi_ch_ind: Mapping[str, int]) -> np.ndarray:
    """One-hot protein: shape ``(max_seq_len, vocab)``. Same index-minus-one rule as SMILES."""
    x = np.zeros((max_seq_len, len(smi_ch_ind)), dtype=np.float32)
    for i, ch in enumerate(line[:max_seq_len]):
        if ch in smi_ch_ind:
            x[i, smi_ch_ind[ch] - 1] = 1.0
    return x


def label_smiles(line: str, max_smi_len: int, smi_ch_ind: Mapping[str, int]) -> np.ndarray:
    """Integer SMILES: shape ``(max_smi_len,)``. Pad / unknown stay 0; known chars keep charset ids."""
    x = np.zeros(max_smi_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_smi_len]):
        if ch in smi_ch_ind:
            x[i] = smi_ch_ind[ch]
    return x


def label_sequence(line: str, max_seq_len: int, smi_ch_ind: Mapping[str, int]) -> np.ndarray:
    """Integer protein: shape ``(max_seq_len,)``. Same padding rule as ``label_smiles``."""
    x = np.zeros(max_seq_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_seq_len]):
        if ch in smi_ch_ind:
            x[i] = smi_ch_ind[ch]
    return x


def encode_smiles(line: str, max_smi_len: int, with_label: bool = True) -> np.ndarray:
    """Default path is integer labels (CNN-CNN + Embedding). One-hot is ``combined_onehot`` only."""
    if with_label:
        return label_smiles(line, max_smi_len, CHARISOSMISET)
    return one_hot_smiles(line, max_smi_len, CHARISOSMISET)


def encode_protein(line: str, max_seq_len: int, with_label: bool = True) -> np.ndarray:
    if with_label:
        return label_sequence(line, max_seq_len, CHARPROTSET)
    return one_hot_sequence(line, max_seq_len, CHARPROTSET)


def log_transform_kd(y: np.ndarray) -> np.ndarray:
    """Davis pKd transform used in the paper: pKd = -log10(Kd / 1e9)."""
    return -(np.log10(y / math.pow(10, 9)))


def encode_all(
    smiles_list: Sequence[str],
    protein_list: Sequence[str],
    max_smi_len: int,
    max_seq_len: int,
    with_label: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode unique drugs/proteins once. Returns ``XD (n_drugs, ...)``, ``XT (n_proteins, ...)``."""
    drugs = [encode_smiles(s, max_smi_len, with_label=with_label) for s in smiles_list]
    proteins = [encode_protein(p, max_seq_len, with_label=with_label) for p in protein_list]
    return np.asarray(drugs), np.asarray(proteins)
