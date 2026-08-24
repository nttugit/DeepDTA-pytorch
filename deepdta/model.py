"""CNN–CNN DeepDTA in PyTorch (Keras ``build_combined_categorical``)."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepdta.data import CHARISOSMILEN, CHARPROTLEN


def _init_keras_like(module: nn.Module) -> None:
    """Match Keras 2 defaults used by the original DeepDTA builders."""
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)


class CNNEncoder(nn.Module):
    """Embedding → 3× Conv1d (n, 2n, 3n, padding=valid) → global max pool."""

    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int, kernel_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, stride=1, padding=0)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x.amax(dim=-1)


class DeepDTA(nn.Module):
    """SMILES CNN + protein CNN → concat → 1024 → 1024 → 512 → 1."""

    def __init__(
        self,
        smi_vocab_size: int = CHARISOSMILEN + 1,
        seq_vocab_size: int = CHARPROTLEN + 1,
        embed_dim: int = 128,
        num_filters: int = 32,
        smi_kernel: int = 8,
        seq_kernel: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.drug_encoder = CNNEncoder(smi_vocab_size, embed_dim, num_filters, smi_kernel)
        self.protein_encoder = CNNEncoder(seq_vocab_size, embed_dim, num_filters, seq_kernel)
        in_dim = num_filters * 3 * 2
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        _init_keras_like(self)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.05)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.drug_encoder(drug), self.protein_encoder(protein)], dim=-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.out(x).squeeze(-1)
