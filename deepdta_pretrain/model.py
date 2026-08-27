"""MLP head over frozen ESM-2 (protein) and ChemBERTa (drug) embeddings."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from deepdta.model import _init_keras_like
from deepdta_pretrain.embeddings import ENCODERS


class DeepDTAPretrain(nn.Module):
    """Frozen PLM vectors → LayerNorm → optional projection → concat → MLP → affinity.

    The head keeps the DeepDTA shape (1024 → 1024 → 512 → 1) so results are
    comparable against the CNN–CNN baseline in ``deepdta/``.
    """

    def __init__(
        self,
        drug_dim: int = ENCODERS["drug"]["dim"],
        prot_dim: int = ENCODERS["protein"]["dim"],
        proj_dim: int = 256,
        hidden: Sequence[int] = (1024, 1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        if not hidden:
            raise ValueError("hidden must contain at least one layer width")

        # ESM-2 and ChemBERTa activations live on different scales; without this
        # one branch dominates the concatenated input.
        self.drug_norm = nn.LayerNorm(drug_dim)
        self.protein_norm = nn.LayerNorm(prot_dim)

        if proj_dim and proj_dim > 0:
            self.drug_proj = nn.Linear(drug_dim, proj_dim)
            self.protein_proj = nn.Linear(prot_dim, proj_dim)
            in_dim = proj_dim * 2
        else:
            self.drug_proj = None
            self.protein_proj = None
            in_dim = drug_dim + prot_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for i, width in enumerate(hidden):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout))
            prev = width
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)
        self.relu = nn.ReLU()

        _init_keras_like(self)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.05)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        d = self.drug_norm(drug)
        p = self.protein_norm(protein)
        if self.drug_proj is not None:
            d = self.relu(self.drug_proj(d))
            p = self.relu(self.protein_proj(p))
        x = self.mlp(torch.cat([d, p], dim=-1))
        return self.out(x).squeeze(-1)
