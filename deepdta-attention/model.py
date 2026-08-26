"""DeepDTA + bidirectional cross-attention (SMILES ↔ protein).

CHANGE vs original CNN–CNN (`deepdta/model.py`):
- CNN encoders no longer global-pool immediately. They keep a feature map
  (B, C, L') so tokens can attend across modalities.
- A bidirectional cross-attention block lets every SMILES position attend to
  protein residues and every residue attend to SMILES tokens (PyTorch
  ``nn.MultiheadAttention`` ≡ Keras ``MultiHeadAttention``).
- Residual + LayerNorm, then the original global-max-pool → concat → FC head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data import CHARISOSMILEN, CHARPROTLEN


def _init_keras_like(module: nn.Module) -> None:
    """Match Keras 2 defaults used by the original DeepDTA CNN/FC layers."""
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


def _valid_heads(embed_dim: int, requested: int) -> int:
    """``nn.MultiheadAttention`` requires embed_dim % num_heads == 0."""
    requested = max(1, int(requested))
    if embed_dim % requested == 0:
        return requested
    for heads in range(min(requested, embed_dim), 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


class CNNEncoder(nn.Module):
    """Embedding → 3× Conv1d (n, 2n, 3n, padding=valid). Returns a sequence map.

    CHANGE: original encoder returned ``amax`` over length. We keep (B, 3n, L')
    so cross-attention can mix SMILES and protein positions. Pooling happens
    after attention in ``DeepDTAAttention``.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_conv = 3
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, stride=1, padding=0)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()

    def _shrink_pad_mask(self, pad_mask: torch.Tensor) -> torch.Tensor:
        """Drop trailing positions removed by three valid convolutions.

        ``pad_mask`` is True on padding (index 0). Valid Conv1d of kernel k
        drops the last (k-1) steps each layer, so 3 layers drop 3*(k-1).
        """
        drop = self.n_conv * (self.kernel_size - 1)
        if drop <= 0:
            return pad_mask
        if pad_mask.size(1) <= drop:
            return pad_mask.new_ones(pad_mask.size(0), 1, dtype=torch.bool)
        return pad_mask[:, : pad_mask.size(1) - drop]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_mask = x == 0
        h = self.embedding(x).transpose(1, 2)
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        return h, self._shrink_pad_mask(pad_mask)


class BidirectionalCrossAttention(nn.Module):
    """SMILES queries protein keys/values and vice versa.

    drug_seq:  Q = SMILES, K/V = protein  → each ligand token sees residues
    prot_seq:  Q = protein, K/V = SMILES  → each residue sees ligand tokens
    Residual + LayerNorm (Transformer-style). Padding positions are masked.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = _valid_heads(dim, n_heads)
        self.drug_attend_prot = nn.MultiheadAttention(
            dim, self.n_heads, dropout=dropout, batch_first=True
        )
        self.prot_attend_drug = nn.MultiheadAttention(
            dim, self.n_heads, dropout=dropout, batch_first=True
        )
        self.norm_drug = nn.LayerNorm(dim)
        self.norm_prot = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        drug: torch.Tensor,
        protein: torch.Tensor,
        drug_pad: torch.Tensor,
        prot_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Conv maps are (B, C, L); MHA wants (B, L, C).
        drug_t = drug.transpose(1, 2)
        prot_t = protein.transpose(1, 2)
        drug_ctx, _ = self.drug_attend_prot(
            drug_t, prot_t, prot_t, key_padding_mask=prot_pad, need_weights=False
        )
        prot_ctx, _ = self.prot_attend_drug(
            prot_t, drug_t, drug_t, key_padding_mask=drug_pad, need_weights=False
        )
        drug_t = self.norm_drug(drug_t + self.dropout(drug_ctx))
        prot_t = self.norm_prot(prot_t + self.dropout(prot_ctx))
        return drug_t.transpose(1, 2), prot_t.transpose(1, 2)


class DeepDTAAttention(nn.Module):
    """CNN(SMILES) + CNN(protein) → cross-attention → pool → 1024 → 1024 → 512 → 1."""

    def __init__(
        self,
        smi_vocab_size: int = CHARISOSMILEN + 1,
        seq_vocab_size: int = CHARPROTLEN + 1,
        embed_dim: int = 128,
        num_filters: int = 32,
        smi_kernel: int = 8,
        seq_kernel: int = 12,
        dropout: float = 0.1,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.drug_encoder = CNNEncoder(smi_vocab_size, embed_dim, num_filters, smi_kernel)
        self.protein_encoder = CNNEncoder(seq_vocab_size, embed_dim, num_filters, seq_kernel)
        conv_dim = num_filters * 3
        self.cross_attn = BidirectionalCrossAttention(conv_dim, attn_heads, attn_dropout)
        in_dim = conv_dim * 2
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
        drug_seq, drug_pad = self.drug_encoder(drug)
        prot_seq, prot_pad = self.protein_encoder(protein)
        drug_seq, prot_seq = self.cross_attn(drug_seq, prot_seq, drug_pad, prot_pad)
        # Same pooling + FC head as original DeepDTA, but on attended maps.
        x = torch.cat([drug_seq.amax(dim=-1), prot_seq.amax(dim=-1)], dim=-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.out(x).squeeze(-1)
