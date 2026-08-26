#!/usr/bin/env python3
"""Download Davis / KIBA files needed by ``python -m deepdta``.

Default destination is ``<repo>/data/{kiba,davis}/``. On Cheaha you can point
``--dest`` at ``$USER_DATA/deepdta/data`` and pass that folder to ``--data-dir``.
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"
FILES = (
    "ligands_can.txt",
    "proteins.txt",
    "Y",
    "folds/test_fold_setting1.txt",
    "folds/train_fold_setting1.txt",
)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "deepdta-cheaha"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=120) as response:
        dest.write_bytes(response.read())


def download_dataset(name: str, dest_root: Path, overwrite: bool) -> None:
    out_dir = dest_root / name
    for rel in FILES:
        dest = out_dir / rel
        if dest.exists() and dest.stat().st_size > 0 and not overwrite:
            print(f"skip {dest}")
            continue
        url = f"{BASE}/{name}/{rel}"
        print(f"get  {url}")
        _fetch(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--datasets", nargs="+", default=["kiba", "davis"], choices=["kiba", "davis"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dest = Path(args.dest)
    for name in args.datasets:
        download_dataset(name, dest, args.overwrite)
    print("done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        sys.exit(1)
