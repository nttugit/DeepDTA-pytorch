"""Prepare a custom dataset in the original DeepDTA-toy format.

Expected files in ``fpath``:

* ``ligands.tab``: header + tab-separated ``id\\tSMILES``
* ``proteins.fasta``: FASTA sequences, one record per protein
* ``Y.tab`` (optional): tab-separated affinity matrix (drugs x proteins).
  Missing values may be written as ``nan``. If omitted, a zero matrix is used.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np

PROT_FILE = "proteins.fasta"
CHEM_FILE = "ligands.tab"
AFF_FILE = "Y.tab"


def prepare_new_data(fpath: Union[str, Path], test: bool = True) -> None:
    fpath = str(fpath)
    if not fpath.endswith(os.sep):
        fpath += os.sep

    proteins = read_proteins(fpath)
    chemicals = read_chemicals(fpath)
    y = np.zeros((len(chemicals), len(proteins)), dtype=np.float64)

    if os.path.exists(fpath + AFF_FILE):
        y = np.loadtxt(fpath + AFF_FILE)

    pickle.dump(y, open(fpath + "Y", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    label_row_inds, _label_col_inds = np.where(np.isnan(y) == False)  # noqa: E712
    indic = list(range(len(label_row_inds)))

    fold_dir = fpath + "folds/"
    os.makedirs(fold_dir, exist_ok=True)
    if test:
        json.dump(indic, open(fold_dir + "test_fold.txt", "w"))
    else:
        json.dump(indic, open(fold_dir + "train_fold.txt", "w"))


def read_chemicals(datafolder: str) -> dict[str, str]:
    chemicals: dict[str, str] = {}
    with open(datafolder + CHEM_FILE, encoding="utf-8") as handle:
        next(handle)
        for row in handle:
            parts = row.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            chemicals[parts[0]] = parts[1].strip()
    print("%d number(s) of chemical(s)" % len(chemicals))
    json.dump(chemicals, open(datafolder + "ligands.txt", "w"))
    return chemicals


def read_proteins(datafolder: str) -> dict[str, str]:
    filename = datafolder + PROT_FILE
    with open(filename, encoding="utf-8") as handle:
        lines = handle.readlines()

    idindex = [i for i, line in enumerate(lines) if line.startswith(">")]
    if lines:
        idindex.append(len(lines))

    proteins: dict[str, str] = {}
    for i, idx in enumerate(idindex[:-1]):
        header = lines[idx].strip()
        pid = _fasta_id(header)
        seq = "".join(lines[idx + 1 : idindex[i + 1]]).replace("\n", "").replace(" ", "")
        proteins[pid] = seq

    print("%d number(s) of protein(s)" % len(proteins))
    json.dump(proteins, open(datafolder + "proteins.txt", "w"))
    return proteins


def _fasta_id(header: str) -> str:
    token = header[1:].split()[0]
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return token
