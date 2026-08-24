"""Lightweight PyTorch DeepDTA (CNN–CNN). Paper: Öztürk et al., Bioinformatics 2018."""

from deepdta.model import DeepDTA
from deepdta.data import load_dataset, make_loader, paper_splits

__all__ = ["DeepDTA", "load_dataset", "make_loader", "paper_splits"]
