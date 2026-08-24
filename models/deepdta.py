"""PyTorch DeepDTA models corresponding to the original Keras builders."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.charset import CHARISOSMILEN, CHARPROTLEN


def keras_glorot_uniform(tensor: torch.Tensor) -> None:
    nn.init.xavier_uniform_(tensor)


def keras_normal(tensor: torch.Tensor, stddev: float = 0.05) -> None:
    nn.init.normal_(tensor, mean=0.0, std=stddev)


def init_keras_like(module: nn.Module) -> None:
    """Approximate Keras 2.x default initializers used by DeepDTA."""
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            keras_glorot_uniform(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            keras_glorot_uniform(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            keras_glorot_uniform(m.weight)


class CNNEncoder(nn.Module):
    """Three Conv1D layers (n, 2n, 3n filters) + global max pooling.

    Parameters
    ----------
    num_filters
        First-layer width ``n`` (paper: 32). Later layers are ``2n`` then ``3n``.
    kernel_size
        Conv window; SMILES and protein encoders use different values.
    vocab_size
        ``CHARLEN + 1`` so index 0 (padding) has an embedding row.
    use_embedding
        True = integer input ``(B, L)``. False = one-hot ``(B, L, C)``.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        kernel_size: int,
        vocab_size: Optional[int] = None,
        embedding_dim: int = 128,
        use_embedding: bool = True,
    ):
        super().__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            if vocab_size is None:
                raise ValueError("vocab_size is required when use_embedding=True")
            self.embedding = nn.Embedding(vocab_size, embedding_dim)  # pad id 0 is *not* masked
            conv_in = embedding_dim
        else:
            self.embedding = None
            conv_in = in_channels

        # padding=0 == Keras padding='valid' → length shrinks by kernel_size-1 each layer
        self.conv1 = nn.Conv1d(conv_in, num_filters, kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, stride=1, padding=0)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_embedding:
            x = self.embedding(x)  # (B, L, E)
            x = x.transpose(1, 2)  # Conv1d wants (B, C, L)
        else:
            # one-hot tensors are stored as (B, L, C)
            x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, 3n)
        return x


class FullyConnectedHead(nn.Module):
    """Regression MLP: 1024 → dropout → 1024 → dropout → 512 → 1. Output is ``(B,)``."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (1024, 1024, 512), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for i, dim in enumerate(hidden):
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            if i < len(hidden) - 1:  # paper: no dropout after the 512 layer
                layers.append(nn.Dropout(dropout))
            prev = dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(x)).squeeze(-1)

    def init_output_normal(self) -> None:
        # Keras Dense(..., kernel_initializer='normal') on the final unit
        keras_normal(self.out.weight, stddev=0.05)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)


class DeepDTA(nn.Module):
    """Main paper model: CNN(SMILES) + CNN(protein) → concat → FC.

    ``combined_categorical`` uses Embedding; ``combined_onehot`` skips it.
    Encoder output is ``3n`` each, so the head sees ``6n`` (192 when n=32).
    """

    def __init__(
        self,
        smi_vocab_size: int = CHARISOSMILEN + 1,  # +1 for pad index 0
        seq_vocab_size: int = CHARPROTLEN + 1,
        embedding_dim: int = 128,
        num_filters: int = 32,
        smi_filter_length: int = 8,  # conv kernel on SMILES
        seq_filter_length: int = 12,  # conv kernel on protein
        dropout: float = 0.1,
        fc_dims: Sequence[int] = (1024, 1024, 512),
        use_embedding: bool = True,
        keras_init: bool = True,
    ):
        super().__init__()
        smi_in = CHARISOSMILEN if not use_embedding else embedding_dim
        seq_in = CHARPROTLEN if not use_embedding else embedding_dim
        self.drug_encoder = CNNEncoder(
            in_channels=smi_in,
            num_filters=num_filters,
            kernel_size=smi_filter_length,
            vocab_size=smi_vocab_size,
            embedding_dim=embedding_dim,
            use_embedding=use_embedding,
        )
        self.protein_encoder = CNNEncoder(
            in_channels=seq_in,
            num_filters=num_filters,
            kernel_size=seq_filter_length,
            vocab_size=seq_vocab_size,
            embedding_dim=embedding_dim,
            use_embedding=use_embedding,
        )
        self.head = FullyConnectedHead(num_filters * 3 * 2, fc_dims, dropout)  # 96+96=192
        if keras_init:
            init_keras_like(self)
            self.head.init_output_normal()

    def forward(self, drug: torch.Tensor, protein: torch.Tensor, **_unused) -> torch.Tensor:
        # ``_unused`` swallows drug_idx / similarity tensors so one trainer loop fits all models
        drug_repr = self.drug_encoder(drug)
        prot_repr = self.protein_encoder(protein)
        return self.head(torch.cat([drug_repr, prot_repr], dim=-1))


class DeepDTASingleDrug(nn.Module):
    """CNN on SMILES + protein identity one-hot (``build_single_drug``)."""

    def __init__(
        self,
        n_proteins: int,
        smi_vocab_size: int = CHARISOSMILEN + 1,
        embedding_dim: int = 128,
        num_filters: int = 32,
        smi_filter_length: int = 8,
        dropout: float = 0.1,
        fc_dims: Sequence[int] = (1024, 1024, 512),
        keras_init: bool = True,
    ):
        super().__init__()
        self.drug_encoder = CNNEncoder(
            in_channels=embedding_dim,
            num_filters=num_filters,
            kernel_size=smi_filter_length,
            vocab_size=smi_vocab_size,
            embedding_dim=embedding_dim,
            use_embedding=True,
        )
        self.head = FullyConnectedHead(num_filters * 3 + n_proteins, fc_dims, dropout)
        if keras_init:
            init_keras_like(self)
            self.head.init_output_normal()

    def forward(self, drug: torch.Tensor, protein_idx: torch.Tensor, n_proteins: int, **_unused) -> torch.Tensor:
        drug_repr = self.drug_encoder(drug)
        prot_oh = F.one_hot(protein_idx, num_classes=n_proteins).float()
        return self.head(torch.cat([drug_repr, prot_oh], dim=-1))


class DeepDTASingleProtein(nn.Module):
    """Drug identity one-hot + CNN on protein (``build_single_prot``)."""

    def __init__(
        self,
        n_drugs: int,
        seq_vocab_size: int = CHARPROTLEN + 1,
        embedding_dim: int = 128,
        num_filters: int = 32,
        seq_filter_length: int = 12,
        dropout: float = 0.1,
        fc_dims: Sequence[int] = (1024, 1024, 512),
        keras_init: bool = True,
    ):
        super().__init__()
        self.protein_encoder = CNNEncoder(
            in_channels=embedding_dim,
            num_filters=num_filters,
            kernel_size=seq_filter_length,
            vocab_size=seq_vocab_size,
            embedding_dim=embedding_dim,
            use_embedding=True,
        )
        self.head = FullyConnectedHead(n_drugs + num_filters * 3, fc_dims, dropout)
        if keras_init:
            init_keras_like(self)
            self.head.init_output_normal()

    def forward(self, protein: torch.Tensor, drug_idx: torch.Tensor, n_drugs: int, **_unused) -> torch.Tensor:
        prot_repr = self.protein_encoder(protein)
        drug_oh = F.one_hot(drug_idx, num_classes=n_drugs).float()
        return self.head(torch.cat([drug_oh, prot_repr], dim=-1))


class DeepDTAIdentityBaseline(nn.Module):
    """Identity vectors compressed then FC (``build_baseline``)."""

    def __init__(
        self,
        n_drugs: int,
        n_proteins: int,
        dropout: float = 0.1,
        fc_dims: Sequence[int] = (1024, 1024, 512),
        keras_init: bool = True,
    ):
        super().__init__()
        self.drug_proj = nn.Linear(n_drugs, 1)
        self.prot_proj = nn.Linear(n_proteins, 1)
        self.head = FullyConnectedHead(2, fc_dims, dropout)
        if keras_init:
            init_keras_like(self)
            self.head.init_output_normal()

    def forward(self, drug_idx: torch.Tensor, protein_idx: torch.Tensor, n_drugs: int, n_proteins: int, **_unused) -> torch.Tensor:
        drug_oh = F.one_hot(drug_idx, num_classes=n_drugs).float()
        prot_oh = F.one_hot(protein_idx, num_classes=n_proteins).float()
        return self.head(torch.cat([self.drug_proj(drug_oh), self.prot_proj(prot_oh)], dim=-1))


class DeepDTASimilarity(nn.Module):
    """S-W + Pubchem similarity rows into the FC head (paper variant)."""

    def __init__(
        self,
        n_drugs: int,
        n_proteins: int,
        dropout: float = 0.1,
        fc_dims: Sequence[int] = (1024, 1024, 512),
        keras_init: bool = True,
    ):
        super().__init__()
        self.head = FullyConnectedHead(n_drugs + n_proteins, fc_dims, dropout)
        if keras_init:
            init_keras_like(self)
            self.head.init_output_normal()

    def forward(self, drug_sim: torch.Tensor, protein_sim: torch.Tensor, **_unused) -> torch.Tensor:
        return self.head(torch.cat([drug_sim, protein_sim], dim=-1))


class DeepDTACNNSimilarityProtein(nn.Module):
    """CNN on SMILES + S-W protein similarity (paper: CNN compounds, S-W proteins)."""

    def __init__(
        self,
        n_proteins: int,
        smi_vocab_size: int = CHARISOSMILEN + 1,
        embedding_dim: int = 128,
        num_filters: int = 32,
        smi_filter_length: int = 8,
        dropout: float = 0.1,
        fc_dims: Sequence[int] = (1024, 1024, 512),
        keras_init: bool = True,
    ):
        super().__init__()
        self.drug_encoder = CNNEncoder(
            in_channels=embedding_dim,
            num_filters=num_filters,
            kernel_size=smi_filter_length,
            vocab_size=smi_vocab_size,
            embedding_dim=embedding_dim,
            use_embedding=True,
        )
        self.head = FullyConnectedHead(num_filters * 3 + n_proteins, fc_dims, dropout)
        if keras_init:
            init_keras_like(self)
            self.head.init_output_normal()

    def forward(self, drug: torch.Tensor, protein_sim: torch.Tensor, **_unused) -> torch.Tensor:
        return self.head(torch.cat([self.drug_encoder(drug), protein_sim], dim=-1))


class DeepDTASimilarityCNNProtein(nn.Module):
    """Pubchem drug similarity + CNN on protein (paper: Pubchem compounds, CNN proteins)."""

    def __init__(
        self,
        n_drugs: int,
        seq_vocab_size: int = CHARPROTLEN + 1,
        embedding_dim: int = 128,
        num_filters: int = 32,
        seq_filter_length: int = 12,
        dropout: float = 0.1,
        fc_dims: Sequence[int] = (1024, 1024, 512),
        keras_init: bool = True,
    ):
        super().__init__()
        self.protein_encoder = CNNEncoder(
            in_channels=embedding_dim,
            num_filters=num_filters,
            kernel_size=seq_filter_length,
            vocab_size=seq_vocab_size,
            embedding_dim=embedding_dim,
            use_embedding=True,
        )
        self.head = FullyConnectedHead(n_drugs + num_filters * 3, fc_dims, dropout)
        if keras_init:
            init_keras_like(self)
            self.head.init_output_normal()

    def forward(self, protein: torch.Tensor, drug_sim: torch.Tensor, **_unused) -> torch.Tensor:
        return self.head(torch.cat([drug_sim, self.protein_encoder(protein)], dim=-1))


# YAML ``model.name`` values. combined_categorical is the paper CNN-CNN; the rest are ablations.
MODEL_REGISTRY = {
    "combined_categorical": "combined_categorical",
    "combined_onehot": "combined_onehot",
    "single_drug": "single_drug",
    "single_prot": "single_prot",
    "baseline": "baseline",
    "similarity": "similarity",
    "cnn_sw": "cnn_sw",
    "pubchem_cnn": "pubchem_cnn",
}


def build_model(
    name: str,
    num_filters: int,
    smi_filter_length: int,
    seq_filter_length: int,
    n_drugs: int,
    n_proteins: int,
    embedding_dim: int = 128,
    dropout: float = 0.1,
    fc_dims: Sequence[int] = (1024, 1024, 512),
    keras_init: bool = True,
    smi_vocab_size: int = CHARISOSMILEN + 1,
    seq_vocab_size: int = CHARPROTLEN + 1,
) -> nn.Module:
    """Factory keyed by ``model.name`` in YAML. Default paper model is ``combined_categorical``."""
    name = name.lower()
    common = dict(
        dropout=dropout,
        fc_dims=fc_dims,
        keras_init=keras_init,
    )
    if name == "combined_categorical":
        return DeepDTA(
            smi_vocab_size=smi_vocab_size,
            seq_vocab_size=seq_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            smi_filter_length=smi_filter_length,
            seq_filter_length=seq_filter_length,
            use_embedding=True,
            **common,
        )
    if name == "combined_onehot":
        return DeepDTA(
            smi_vocab_size=smi_vocab_size,
            seq_vocab_size=seq_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            smi_filter_length=smi_filter_length,
            seq_filter_length=seq_filter_length,
            use_embedding=False,
            **common,
        )
    if name == "single_drug":
        return DeepDTASingleDrug(
            n_proteins=n_proteins,
            smi_vocab_size=smi_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            smi_filter_length=smi_filter_length,
            **common,
        )
    if name == "single_prot":
        return DeepDTASingleProtein(
            n_drugs=n_drugs,
            seq_vocab_size=seq_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            seq_filter_length=seq_filter_length,
            **common,
        )
    if name in {"baseline", "identity_baseline"}:
        return DeepDTAIdentityBaseline(n_drugs=n_drugs, n_proteins=n_proteins, **common)
    if name == "similarity":
        return DeepDTASimilarity(n_drugs=n_drugs, n_proteins=n_proteins, **common)
    if name in {"cnn_sw", "cnn_similarity_protein"}:
        return DeepDTACNNSimilarityProtein(
            n_proteins=n_proteins,
            smi_vocab_size=smi_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            smi_filter_length=smi_filter_length,
            **common,
        )
    if name in {"pubchem_cnn", "similarity_cnn_protein"}:
        return DeepDTASimilarityCNNProtein(
            n_drugs=n_drugs,
            seq_vocab_size=seq_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            seq_filter_length=seq_filter_length,
            **common,
        )
    raise ValueError(f"Unknown model name: {name}")
