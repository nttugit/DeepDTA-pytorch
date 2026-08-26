#!/usr/bin/env python3
"""Write a tiny DeepDTA-format dataset under dummy_data/ for smoke tests."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "dummy_data"

# Lengths stay above 3*(kernel-1)=9 so three valid Conv1d layers (k=4) still have a time axis.
LIGANDS = {
    "D1": "CCOC1=CC=CC=C1C(=O)O",
    "D2": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "D3": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "D4": "NC1=NC=NC2=C1N=CN2C3OC(CO)C(O)C3O",
    "D5": "CC1=C(C=CC=C1)NC2=NC=CC=N2",
    "D6": "C1=CC=C(C=C1)C2=CC=CC=C2",
    "D7": "CCC1=CC=C(C=C1)C(=O)O",
    "D8": "COC1=CC=C(C=C1)CCN",
}

PROTEINS = {
    "P1": "MKKFFDSRREQGGSGLGSGSSGGGGSGGGYT",
    "P2": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQ",
    "P3": "GSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSMEEPQSDPSVEPPLSQETFSDL",
    "P4": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVAT",
    "P5": "APLALTQIGKIVQETLEKLEKAAEAAEAAEAAKAAKAAKGGGSGGGSGGGS",
    "P6": "MSEIDRISNELKAMVAALRQEGLDPDQADQEAKLKEKLEKLLKEKEKLLKEK",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "folds").mkdir(exist_ok=True)

    (OUT / "ligands_can.txt").write_text(json.dumps(LIGANDS, indent=2), encoding="utf-8")
    (OUT / "proteins.txt").write_text(json.dumps(PROTEINS, indent=2), encoding="utf-8")

    rng = np.random.default_rng(1)
    n_d, n_p = len(LIGANDS), len(PROTEINS)
    y = rng.normal(loc=7.2, scale=1.1, size=(n_d, n_p))
    # A few missing pairs, same as real KIBA/Davis Y matrices.
    y[0, 5] = np.nan
    y[7, 0] = np.nan
    y[3, 2] = np.nan
    y[5, 4] = np.nan
    with (OUT / "Y").open("wb") as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Fold files index the *labeled-pair list* (len = non-NaN cells), not Y.ravel().
    n_labeled = int(np.isfinite(y).sum())
    labeled = list(range(n_labeled))
    test = labeled[-8:]
    rest = labeled[:-8]
    mid = len(rest) // 2
    train_folds = [rest[:mid], rest[mid:]]
    (OUT / "folds" / "test_fold.txt").write_text(json.dumps(test), encoding="utf-8")
    (OUT / "folds" / "train_fold.txt").write_text(json.dumps(train_folds), encoding="utf-8")

    print(f"Wrote {OUT}")
    print(f"drugs={n_d} proteins={n_p} labeled={len(labeled)} test={len(test)} train_folds={[len(f) for f in train_folds]}")


if __name__ == "__main__":
    main()
