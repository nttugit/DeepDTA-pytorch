"""Frozen PLM embeddings for drugs (ChemBERTa) and proteins (ESM-2), cached to disk.

Davis/KIBA have few unique entities (68/2111 drugs, 442/229 proteins) but tens of
thousands of pairs, so every drug and protein is encoded exactly once and the
training loop only reads vectors.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

# Tokenizing before the DataLoader forks triggers a noisy warning otherwise.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "cache" / "embeddings"

# `max_len` counts content tokens; special tokens are added on top of it.
# ESM-2 has max_position_embeddings=1026, ChemBERTa (RoBERTa) has 515.
ENCODERS: dict[str, dict[str, Any]] = {
    "drug": {
        "hf_name": "DeepChem/ChemBERTa-77M-MLM",
        "dim": 384,
        "max_len": 512,
    },
    "protein": {
        "hf_name": "facebook/esm2_t12_35M_UR50D",
        "dim": 480,
        "max_len": 1022,
    },
}

DeviceLike = Union[torch.device, str, None]


def encoder_defaults(kind: str) -> dict[str, Any]:
    if kind not in ENCODERS:
        raise ValueError(f"Unknown encoder kind {kind!r}. Choose from {list(ENCODERS)}")
    return dict(ENCODERS[kind])


def _resolve_device(device: DeviceLike) -> torch.device:
    if isinstance(device, torch.device):
        return device
    name = (device or "auto").lower()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" or (name == "auto" and has_mps):
        return torch.device("mps")
    if name in {"auto", "cpu"}:
        return torch.device("cpu")
    return torch.device(name)


def mean_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average token states, ignoring padding and CLS/EOS/SEP."""
    mask = attention_mask.to(hidden.dtype)
    if special_tokens_mask is not None:
        mask = mask * (1.0 - special_tokens_mask.to(hidden.dtype))
    mask = mask.unsqueeze(-1)
    total = (hidden * mask).sum(dim=1)
    return total / mask.sum(dim=1).clamp(min=1.0)


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0]


def content_length_cap(config: Any, tokenizer: Any, n_special: int) -> Optional[int]:
    """Largest number of content tokens the model can actually embed.

    ChemBERTa (RoBERTa) is the reason this exists: its position embedding table
    holds 515 rows but RoBERTa offsets position ids by ``pad_token_id + 1``, so a
    514-token input indexes row 515 and raises ``IndexError: index out of range
    in self``. ESM-2 uses rotary embeddings and is not bounded this way.
    """
    caps: list[int] = []

    limit = getattr(config, "max_position_embeddings", None)
    if limit:
        if getattr(config, "position_embedding_type", "absolute") == "absolute":
            limit -= (getattr(config, "pad_token_id", None) or 0) + 1
        caps.append(int(limit))

    # ESM sets this to a sentinel (~1e30) meaning "unbounded"; ignore that.
    model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 0 < model_max < 1_000_000:
        caps.append(model_max)

    if not caps:
        return None
    return max(1, min(caps) - n_special)


def encode_texts(
    texts: Sequence[str],
    hf_name: str,
    max_len: int,
    pool: str = "mean",
    device: DeviceLike = "auto",
    batch_size: int = 8,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a frozen HuggingFace encoder over `texts` and pool to one vector each.

    Returns the `[N, dim]` float32 embeddings plus stats (token lengths, how many
    inputs were truncated).
    """
    from transformers import AutoModel, AutoTokenizer
    from transformers import logging as hf_logging

    if pool not in {"mean", "cls"}:
        raise ValueError(f"Unknown pool {pool!r}. Choose from ['mean', 'cls']")

    # The MLM checkpoints have no pooler; we never use it, so skip that warning.
    hf_logging.set_verbosity_error()
    torch_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name).to(torch_device)
    model.eval()

    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    requested_max_len = int(max_len)
    cap = content_length_cap(model.config, tokenizer, n_special)
    if cap is not None and max_len > cap:
        print(
            f"[deepdta_pretrain] {hf_name} supports at most {cap} content tokens; "
            f"clamping max_len {max_len} -> {cap}"
        )
        max_len = cap
    max_length = max_len + n_special

    # One untruncated pass: exact token lengths for stats and length-sorted batching.
    raw_lengths = np.asarray(
        [
            len(ids)
            for ids in tokenizer(
                list(texts), add_special_tokens=True, truncation=False, verbose=False
            )["input_ids"]
        ],
        dtype=np.int64,
    )
    n_truncated = int((raw_lengths > max_length).sum())

    order = np.argsort(-raw_lengths, kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)

    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    iterator = tqdm(batches, desc=f"encode {hf_name}") if show_progress else batches
    for batch_idx in iterator:
        batch = [texts[i] for i in batch_idx]
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = enc.pop("special_tokens_mask")
        enc = {k: v.to(torch_device) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
        if pool == "mean":
            vec = mean_pool(hidden, enc["attention_mask"], special.to(torch_device))
        else:
            vec = cls_pool(hidden)
        out[batch_idx] = vec.float().cpu().numpy()

    stats = {
        "hf_name": hf_name,
        "dim": int(model.config.hidden_size),
        "pool": pool,
        "max_len": int(max_len),
        "max_len_requested": requested_max_len,
        "max_length_with_special": int(max_length),
        "n_texts": int(len(texts)),
        "n_truncated": n_truncated,
        "token_len_min": int(raw_lengths.min()) if len(raw_lengths) else 0,
        "token_len_max": int(raw_lengths.max()) if len(raw_lengths) else 0,
        "token_len_mean": float(raw_lengths.mean()) if len(raw_lengths) else 0.0,
    }
    del model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return out, stats


def _slug(hf_name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "-", hf_name).strip("-").lower()


def cache_path(
    dataset: str,
    kind: str,
    hf_name: str,
    pool: str,
    max_len: int,
    cache_root: Union[str, Path, None] = None,
) -> Path:
    root = Path(cache_root) if cache_root is not None else CACHE_ROOT
    return root / dataset / f"{kind}_{_slug(hf_name)}_{pool}_{max_len}.npz"


def build_cache(
    path: Path,
    keys: Sequence[str],
    texts: Sequence[str],
    hf_name: str,
    max_len: int,
    pool: str = "mean",
    device: DeviceLike = "auto",
    batch_size: int = 8,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    emb, stats = encode_texts(
        texts,
        hf_name=hf_name,
        max_len=max_len,
        pool=pool,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        emb=emb,
        keys=np.asarray(list(keys), dtype=object),
        meta=json.dumps(stats),
    )
    return emb, stats


def load_cache(path: Path, keys: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a cache and refuse it if the entity order no longer matches."""
    with np.load(path, allow_pickle=True) as handle:
        emb = np.asarray(handle["emb"], dtype=np.float32)
        cached_keys = [str(k) for k in handle["keys"].tolist()]
        meta = json.loads(str(handle["meta"]))
    wanted = [str(k) for k in keys]
    if cached_keys != wanted:
        raise ValueError(
            f"Cache {path} does not match the dataset (cached {len(cached_keys)} keys, "
            f"dataset has {len(wanted)}). Delete the file and re-run precompute."
        )
    if emb.shape[0] != len(wanted):
        raise ValueError(f"Cache {path} has {emb.shape[0]} rows for {len(wanted)} keys.")
    return emb, meta


def get_or_build(
    dataset: str,
    kind: str,
    keys: Sequence[str],
    texts: Sequence[str],
    hf_name: Optional[str] = None,
    max_len: Optional[int] = None,
    pool: str = "mean",
    device: DeviceLike = "auto",
    batch_size: int = 8,
    rebuild: bool = False,
    cache_root: Union[str, Path, None] = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = encoder_defaults(kind)
    hf_name = hf_name or spec["hf_name"]
    max_len = spec["max_len"] if max_len is None else max_len
    path = cache_path(dataset, kind, hf_name, pool, max_len, cache_root)

    if path.exists() and not rebuild:
        return load_cache(path, keys)

    print(f"[deepdta_pretrain] encoding {len(texts)} {kind}s with {hf_name} -> {path}")
    return build_cache(
        path,
        keys=keys,
        texts=texts,
        hf_name=hf_name,
        max_len=max_len,
        pool=pool,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
    )
