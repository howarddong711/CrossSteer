#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch

def describe_vector(path: Path) -> str:
    try:
        vec = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        vec = torch.load(path, map_location="cpu")
    if not torch.is_tensor(vec):
        raise TypeError(f"{path} did not load to a Tensor: {type(vec)!r}")
    flat = vec.float().reshape(-1)
    norm = torch.linalg.vector_norm(flat).item()
    return f"{path}: shape={tuple(vec.shape)} dtype={vec.dtype} norm={norm:.6f}"

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CrossSteer .pt steering vectors.")
    parser.add_argument(
        "root",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "vectors"),
        help="Vector root to scan",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    paths = sorted(root.rglob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt vectors found under {root}")

    for path in paths:
        print(describe_vector(path))
    print(f"checked={len(paths)} root={os.path.abspath(root)}")

if __name__ == "__main__":
    main()
