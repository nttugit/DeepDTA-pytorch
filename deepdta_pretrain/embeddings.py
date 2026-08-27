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

# Alternatives worth A/B testing. `dim` here is documentation only: encode_texts
# reads model.config.hidden_size and data.py derives the MLP width from the cache
# shape, so swapping --drug-model / --protein-model needs no code change.
PRESETS: dict[str, dict[str, Any]] = {
    "esm2-35m": {"kind": "protein", "hf_name": "facebook/esm2_t12_35M_UR50D", "dim": 480},
    "esm2-150m": {"kind": "protein", "hf_name": "facebook/esm2_t30_150M_UR50D", "dim": 640},
    "esm2-650m": {"kind": "protein", "hf_name": "facebook/esm2_t33_650M_UR50D", "dim": 1280},
    "chemberta-mlm": {"kind": "drug", "hf_name": "DeepChem/ChemBERTa-77M-MLM", "dim": 384},
    # Multi-task regression over ~200 molecular properties, so its pooled vector
    # is a molecule-level descriptor rather than an MLM token predictor.
    "chemberta-mtr": {"kind": "drug", "hf_name": "DeepChem/ChemBERTa-77M-MTR", "dim": 384},
}

LONG_STRATEGIES = ("truncate", "window")

DeviceLike = Union[torch.device, str, None]


def resolve_model_name(name: Optional[str], kind: str) -> Optional[str]:
    """Map a preset alias to its HuggingFace id; pass real ids through unchanged."""
    if name is None:
        return None
    preset = PRESETS.get(name.lower())
    if preset is None:
        return name
    if preset["kind"] != kind:
        raise ValueError(f"Preset {name!r} is a {preset['kind']} encoder, not {kind}")
    return preset["hf_name"]


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


def max_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-dimension max over token states, ignoring padding and CLS/EOS/SEP."""
    keep = attention_mask.to(torch.bool)
    if special_tokens_mask is not None:
        keep = keep & ~special_tokens_mask.to(torch.bool)
    keep = keep.unsqueeze(-1)
    filled = hidden.masked_fill(~keep, torch.finfo(hidden.dtype).min)
    pooled = filled.amax(dim=1)
    # An input with no content tokens would otherwise pool to -inf.
    return torch.where(keep.any(dim=1), pooled, torch.zeros_like(pooled))


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0]


POOLS = ("mean", "max", "cls")


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


def _apply_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    special: torch.Tensor,
    pool: str,
) -> torch.Tensor:
    if pool == "mean":
        return mean_pool(hidden, attention_mask, special)
    if pool == "max":
        return max_pool(hidden, attention_mask, special)
    return cls_pool(hidden)


def _encode_truncate(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    torch_device: torch.device,
    max_length: int,
    pool: str,
    batch_size: int,
    show_progress: bool,
    raw_lengths: np.ndarray,
    desc: str,
) -> np.ndarray:
    order = np.argsort(-raw_lengths, kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)

    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    iterator = tqdm(batches, desc=desc) if show_progress else batches
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
        vec = _apply_pool(hidden, enc["attention_mask"], special.to(torch_device), pool)
        out[batch_idx] = vec.float().cpu().numpy()
    return out


def _split_windows(ids: Sequence[int], max_len: int, stride: int) -> list[Sequence[int]]:
    """Overlapping windows that always reach the end of the sequence."""
    if len(ids) <= max_len:
        return [ids]
    chunks: list[Sequence[int]] = []
    start = 0
    while True:
        chunks.append(ids[start : start + max_len])
        if start + max_len >= len(ids):
            return chunks
        start += stride


def _encode_window(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    torch_device: torch.device,
    max_len: int,
    pool: str,
    batch_size: int,
    show_progress: bool,
    desc: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Pool each window separately, then average per text weighted by window length.

    Truncation throws away the tail of every long protein (Davis reaches 2551
    tokens against a 1022 cap), so the residues that distinguish two kinases may
    never be seen. Windowing keeps the whole sequence at a modest encode cost.
    """
    stride = max(1, max_len // 2)
    content = tokenizer(list(texts), add_special_tokens=False, truncation=False, verbose=False)
    all_ids = content["input_ids"]

    chunks: list[Sequence[int]] = []
    owners: list[int] = []
    for i, ids in enumerate(all_ids):
        windows = _split_windows(ids, max_len, stride)
        chunks.extend(windows)
        owners.extend([i] * len(windows))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dim = model.config.hidden_size
    totals = np.zeros((len(texts), dim), dtype=np.float64)
    weights = np.zeros(len(texts), dtype=np.float64)

    order = np.argsort([-len(c) for c in chunks], kind="stable")
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    iterator = tqdm(batches, desc=desc) if show_progress else batches
    for batch_idx in iterator:
        seqs = [tokenizer.build_inputs_with_special_tokens(list(chunks[i])) for i in batch_idx]
        width = max(len(s) for s in seqs)
        input_ids = np.full((len(seqs), width), pad_id, dtype=np.int64)
        attn = np.zeros((len(seqs), width), dtype=np.int64)
        special = np.ones((len(seqs), width), dtype=np.int64)
        for j, seq in enumerate(seqs):
            input_ids[j, : len(seq)] = seq
            attn[j, : len(seq)] = 1
            special[j, : len(seq)] = tokenizer.get_special_tokens_mask(
                seq, already_has_special_tokens=True
            )
        enc = {
            "input_ids": torch.from_numpy(input_ids).to(torch_device),
            "attention_mask": torch.from_numpy(attn).to(torch_device),
        }
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
        vec = _apply_pool(
            hidden, enc["attention_mask"], torch.from_numpy(special).to(torch_device), pool
        )
        vec = vec.float().cpu().numpy().astype(np.float64)
        for j, chunk_idx in enumerate(batch_idx):
            owner = owners[chunk_idx]
            w = float(max(len(chunks[chunk_idx]), 1))
            totals[owner] += vec[j] * w
            weights[owner] += w

    out = (totals / np.maximum(weights, 1.0)[:, None]).astype(np.float32)
    per_text = np.bincount(owners, minlength=len(texts))
    return out, {
        "n_windowed": int((per_text > 1).sum()),
        "n_windows_total": int(len(chunks)),
        "n_windows_max": int(per_text.max()) if len(per_text) else 0,
        "window_stride": int(stride),
    }


def encode_texts(
    texts: Sequence[str],
    hf_name: str,
    max_len: int,
    pool: str = "mean",
    device: DeviceLike = "auto",
    batch_size: int = 8,
    show_progress: bool = True,
    long_strategy: str = "truncate",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a frozen HuggingFace encoder over `texts` and pool to one vector each.

    Returns the `[N, dim]` float32 embeddings plus stats (token lengths, how many
    inputs were truncated or split into windows).
    """
    from transformers import AutoModel, AutoTokenizer
    from transformers import logging as hf_logging

    if pool not in POOLS:
        raise ValueError(f"Unknown pool {pool!r}. Choose from {list(POOLS)}")
    if long_strategy not in LONG_STRATEGIES:
        raise ValueError(
            f"Unknown long_strategy {long_strategy!r}. Choose from {list(LONG_STRATEGIES)}"
        )

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
    desc = f"encode {hf_name}"

    extra: dict[str, Any] = {}
    if long_strategy == "window":
        out, extra = _encode_window(
            texts, tokenizer, model, torch_device, max_len, pool,
            batch_size, show_progress, desc,
        )
        n_truncated = 0
    else:
        out = _encode_truncate(
            texts, tokenizer, model, torch_device, max_length, pool,
            batch_size, show_progress, raw_lengths, desc,
        )

    stats = {
        "hf_name": hf_name,
        "dim": int(model.config.hidden_size),
        "pool": pool,
        "max_len": int(max_len),
        "max_len_requested": requested_max_len,
        "max_length_with_special": int(max_length),
        "long_strategy": long_strategy,
        "n_texts": int(len(texts)),
        "n_truncated": n_truncated,
        "token_len_min": int(raw_lengths.min()) if len(raw_lengths) else 0,
        "token_len_max": int(raw_lengths.max()) if len(raw_lengths) else 0,
        "token_len_mean": float(raw_lengths.mean()) if len(raw_lengths) else 0.0,
        **extra,
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
    long_strategy: str = "truncate",
) -> Path:
    root = Path(cache_root) if cache_root is not None else CACHE_ROOT
    # Omitted for "truncate" so caches built before windowing existed stay valid.
    suffix = "" if long_strategy == "truncate" else f"_{long_strategy}"
    return root / dataset / f"{kind}_{_slug(hf_name)}_{pool}_{max_len}{suffix}.npz"


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
    long_strategy: str = "truncate",
) -> tuple[np.ndarray, dict[str, Any]]:
    emb, stats = encode_texts(
        texts,
        hf_name=hf_name,
        max_len=max_len,
        pool=pool,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
        long_strategy=long_strategy,
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
    long_strategy: str = "truncate",
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = encoder_defaults(kind)
    hf_name = resolve_model_name(hf_name, kind) or spec["hf_name"]
    max_len = spec["max_len"] if max_len is None else max_len
    path = cache_path(dataset, kind, hf_name, pool, max_len, cache_root, long_strategy)

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
        long_strategy=long_strategy,
    )


FINGERPRINTS = ("none", "ecfp4")


def encode_ecfp(
    smiles: Sequence[str],
    n_bits: int = 2048,
    radius: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Morgan/ECFP bit vectors.

    Mean-pooled ChemBERTa blurs substructure identity, which hurts most on KIBA
    (2111 drugs against 229 proteins, so the drug side carries the signal). ECFP
    keeps exact substructure presence and complements the learned vector.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    n_failed = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
            continue
        out[i] = np.asarray(gen.GetFingerprint(mol), dtype=np.float32)

    stats = {
        "fingerprint": f"ecfp{2 * radius}",
        "dim": int(n_bits),
        "radius": int(radius),
        "n_bits": int(n_bits),
        "n_texts": int(len(smiles)),
        "n_failed": n_failed,
        "bits_on_mean": float(out.sum(axis=1).mean()) if len(smiles) else 0.0,
    }
    if n_failed:
        print(f"[deepdta_pretrain] WARNING: RDKit could not parse {n_failed} SMILES")
    return out, stats


def get_or_build_fingerprint(
    dataset: str,
    keys: Sequence[str],
    smiles: Sequence[str],
    n_bits: int = 2048,
    radius: int = 2,
    rebuild: bool = False,
    cache_root: Union[str, Path, None] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    root = Path(cache_root) if cache_root is not None else CACHE_ROOT
    path = root / dataset / f"drug_ecfp{2 * radius}_{n_bits}.npz"
    if path.exists() and not rebuild:
        return load_cache(path, keys)

    print(f"[deepdta_pretrain] fingerprinting {len(smiles)} drugs -> {path}")
    emb, stats = encode_ecfp(smiles, n_bits=n_bits, radius=radius)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        emb=emb,
        keys=np.asarray(list(keys), dtype=object),
        meta=json.dumps(stats),
    )
    return emb, stats
