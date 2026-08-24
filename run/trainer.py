"""Training loop with early stopping, matching the original Keras protocol."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, **_kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, **_kwargs):
            return None

from evaluation.metrics import concordance_index, evaluate_all, mse
from utils import ensure_dir, save_checkpoint


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def model_forward(model: nn.Module, batch: dict[str, torch.Tensor], n_drugs: int, n_proteins: int) -> torch.Tensor:
    """Call any registry model with the full batch; unused kwargs are ignored."""
    return model(
        drug=batch.get("drug"),
        protein=batch.get("protein"),
        drug_idx=batch.get("drug_idx"),
        protein_idx=batch.get("protein_idx"),
        drug_sim=batch.get("drug_sim"),
        protein_sim=batch.get("protein_sim"),
        n_drugs=n_drugs,
        n_proteins=n_proteins,
    )


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_drugs: int,
    n_proteins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return concatenated ``(y_true, y_pred)`` vectors over ``loader``."""
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model_forward(model, batch, n_drugs, n_proteins)
        preds.append(out.detach().cpu().numpy())
        trues.append(batch["affinity"].detach().cpu().numpy())
    return np.concatenate(trues), np.concatenate(preds)


def plot_history(history: dict[str, list[float]], out_path: Union[str, Path]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib/pillow is not installed; skip loss plots")
        return

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    plt.figure()
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["trainloss", "valloss"], loc="upper left")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    acc_path = out_path.with_name(out_path.stem + "_acc" + out_path.suffix)
    plt.figure()
    plt.title("model concordance index")
    plt.ylabel("cindex")
    plt.xlabel("epoch")
    plt.plot(history["ci"])
    plt.plot(history["val_ci"])
    plt.legend(["traincindex", "valcindex"], loc="upper left")
    plt.savefig(acc_path, dpi=120, bbox_inches="tight")
    plt.close()


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_drugs: int,
    n_proteins: int,
    epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    patience: int = 15,  # stop after this many epochs without val-MSE improvement
    restore_best_weights: bool = False,  # paper: False → last.pt is the stopped state, not best
    aupr_threshold: float = 12.1,
    checkpoint_path: Optional[Union[str, Path]] = None,  # writes best.pt; also last.pt beside it
    figure_path: Optional[Union[str, Path]] = None,
    logger=None,
) -> dict[str, Any]:
    """Adam + MSE, early-stop on val MSE. ``best.pt`` tracks lowest val loss; ``last.pt`` is halt state."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = {"loss": [], "val_loss": [], "ci": [], "val_ci": []}
    best_val = float("inf")
    best_state = None
    stale = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        batch_ci = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model_forward(model, batch, n_drugs, n_proteins)
            loss = criterion(pred, batch["affinity"])  # both (B,)
            loss.backward()
            optimizer.step()
            bs = batch["affinity"].size(0)
            running += loss.item() * bs
            n_seen += bs
            n_batches += 1
            batch_ci += concordance_index(
                batch["affinity"].detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
            )
            pbar.set_postfix(loss=f"{running / max(n_seen, 1):.4f}")

        train_loss = running / max(n_seen, 1)
        train_ci = batch_ci / max(n_batches, 1)

        val_true, val_pred = predict_loader(model, val_loader, device, n_drugs, n_proteins)
        val_loss = mse(val_true, val_pred)
        val_ci = concordance_index(val_true, val_pred)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["ci"].append(train_ci)
        history["val_ci"].append(val_ci)

        msg = (
            f"Epoch {epoch:03d} | train MSE {train_loss:.5f} CI {train_ci:.5f} | "
            f"val MSE {val_loss:.5f} CI {val_ci:.5f}"
        )
        if logger is not None:
            logger.log(msg)
        else:
            print(msg)

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                save_checkpoint(
                    {
                        "model_state": best_state,
                        "epoch": epoch,
                        "val_loss": best_val,
                        "val_ci": val_ci,
                    },
                    checkpoint_path,
                )
        else:
            stale += 1
            if stale >= patience:
                stopped_epoch = epoch
                if logger is not None:
                    logger.log(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)

    # last.pt is the halt-time weights (often not the best unless restore_best_weights=True)
    if checkpoint_path is not None:
        save_checkpoint(
            {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": stopped_epoch,
                "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            },
            Path(checkpoint_path).with_name("last.pt"),
        )

    if figure_path is not None:
        plot_history(history, figure_path)

    val_true, val_pred = predict_loader(model, val_loader, device, n_drugs, n_proteins)
    metrics = evaluate_all(val_true, val_pred, aupr_threshold)
    return {
        "history": history,
        "metrics": metrics,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val,
        "y_true": val_true,
        "y_pred": val_pred,
    }
