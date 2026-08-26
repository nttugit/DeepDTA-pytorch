"""DeepDTA-Attention package. Run: python train.py train --dataset dummy"""

from model import DeepDTAAttention
from data import load_dataset, make_loader, paper_splits

__all__ = ["DeepDTAAttention", "load_dataset", "make_loader", "paper_splits"]
