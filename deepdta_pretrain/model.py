"""MLP head over frozen ESM-2 (protein) and ChemBERTa (drug) embeddings."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from deepdta.model import _init_keras_like
from deepdta_pretrain.embeddings import ENCODERS

HIDDEN: tuple[int, ...] = (512, 256)


class DeepDTAPretrain(nn.Module):
    """Pooled PLM vectors → LayerNorm → concat → FC(512) → FC(256) → affinity.

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

        # ESM-2 and ChemBERTa activations live on different scales; normalising
        # each branch on its own keeps one from dominating the concatenation.
        self.drug_norm = nn.LayerNorm(drug_dim)
        self.protein_norm = nn.LayerNorm(prot_dim)

        layers: list[nn.Module] = []
        prev = drug_dim + prot_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = width
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

        # Identity until set_target_stats() is called, so predictions stay in the
        # affinity units the metrics expect either way. Buffers ride along in
        # state_dict, so checkpoints carry the scaling with them.
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))

        _init_keras_like(self)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.05)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def set_target_stats(self, mean: float, std: float) -> None:
        """Let the head regress a standardised target while still emitting affinities.

        Davis puts 70% of its labels on a single censored value, so an unpinned
        output offset drifts between epochs and MSE swings even when the ranking
        (CI) is stable. Fixing mean/std from the training split anchors it.
        """
        std = float(std)
        if not std > 0:
            raise ValueError(f"target std must be positive, got {std}")
        self.target_mean.fill_(float(mean))
        self.target_std.fill_(std)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        d = self.drug_norm(drug)
        p = self.protein_norm(protein)
        x = self.mlp(torch.cat([d, p], dim=-1))
        return self.out(x).squeeze(-1) * self.target_std + self.target_mean
