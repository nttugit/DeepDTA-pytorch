"""MLP head over frozen ESM-2 (protein) and ChemBERTa (drug) embeddings."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from deepdta.model import _init_keras_like
from deepdta_pretrain.embeddings import ENCODERS

HIDDEN: tuple[int, ...] = (512, 256)


class DeepDTAPretrain(nn.Module):
    """Pooled PLM vectors → concat → FC(512) → FC(256) → affinity.

    Both encoders are frozen and pooled upstream, so this only sees one vector
    per drug and one per protein.
    """

    def __init__(
        self,
        drug_dim: int = ENCODERS["drug"]["dim"],
        prot_dim: int = ENCODERS["protein"]["dim"],
        hidden: Sequence[int] = HIDDEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not hidden:
            raise ValueError("hidden must contain at least one layer width")

        layers: list[nn.Module] = []
        prev = drug_dim + prot_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = width
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

        _init_keras_like(self)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.05)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        x = self.mlp(torch.cat([drug, protein], dim=-1))
        return self.out(x).squeeze(-1)
