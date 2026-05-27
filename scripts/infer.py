"""
Score a dataset with the realism classifier.

Accepts either:
  - a .pkl file  (dict with 'embeddings' and optional 'paths' keys)
  - a .npy / .npz file + optional metadata CSV

Usage — pkl (quickstart):
    python scripts/infer.py \
        --input      /path/to/embeddings.pkl \
        --checkpoint outputs/runs/run_03/best_model.pt \
        --output     outputs/scores.csv

Usage — npz/npy + metadata:
    python scripts/infer.py \
        --input      data/embeddings.npz \
        --metadata   data/metadata.csv \
        --checkpoint outputs/runs/run_03/best_model.pt \
        --output     outputs/scores.csv

Note: embeddings must be raw / unscaled DINOv2-base (768-dim) CLS token outputs.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from realism_classifier.config import resolve_device
from realism_classifier.model import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pkl(path):
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict) and "embeddings" in d:
        emb   = d["embeddings"].astype(np.float32)
        paths = d.get("paths", [f"sample_{i}" for i in range(len(emb))])
    elif isinstance(d, dict):
        emb   = np.array(list(d.values())[0]).astype(np.float32)
        paths = [f"sample_{i}" for i in range(len(emb))]
    else:
        emb   = np.array(d).astype(np.float32)
        paths = [f"sample_{i}" for i in range(len(emb))]
    return emb, paths


def load_npy_npz(path, metadata_csv=None):
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(path)
        # use test split if available, otherwise concatenate all
        if "test" in data:
            emb = data["test"].astype(np.float32)
        else:
            emb = np.concatenate([data[k] for k in data.files], axis=0).astype(np.float32)
    else:
        emb = np.load(path).astype(np.float32)

    if metadata_csv:
        meta  = pd.read_csv(metadata_csv)
        paths = meta["sample_id"].tolist() if "sample_id" in meta.columns else [f"sample_{i}" for i in range(len(emb))]
    else:
        paths = [f"sample_{i}" for i in range(len(emb))]
    return emb, paths


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(emb, model, device, batch_size=512):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(emb), batch_size):
            b = torch.from_numpy(emb[i:i + batch_size]).to(device)
            scores.append(torch.sigmoid(model(b).squeeze(-1)).cpu().numpy())
    return np.concatenate(scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score embeddings with the realism classifier.")
    parser.add_argument("--input",      required=True,  help=".pkl / .npy / .npz embeddings file")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pt model checkpoint")
    parser.add_argument("--output",     required=True,  help="Output CSV path")
    parser.add_argument("--metadata",   default=None,   help="Metadata CSV (for .npy/.npz inputs)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device",     default="auto", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    suffix = Path(args.input).suffix.lower()
    print(f"Loading embeddings from {args.input} ...")
    if suffix == ".pkl":
        emb, paths = load_pkl(args.input)
    elif suffix in (".npy", ".npz"):
        emb, paths = load_npy_npz(args.input, args.metadata)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .pkl, .npy, or .npz")

    print(f"  {len(emb):,} samples  dim={emb.shape[1]}  mean={emb.mean():.4f}  std={emb.std():.4f}")

    device = resolve_device(args.device)
    print(f"Loading model from {args.checkpoint} (device={device}) ...")
    model = load_model_from_checkpoint(args.checkpoint, device=device)

    print("Running inference ...")
    scores = run_inference(emb, model, device, batch_size=args.batch_size)

    df = pd.DataFrame({
        "filename":      [os.path.basename(str(p)) for p in paths],
        "path":          paths,
        "realism_score": scores,
    })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nSaved {len(scores):,} scores → {args.output}")
    print(f"  mean={scores.mean():.4f}  median={float(np.median(scores)):.4f}  "
          f"std={scores.std():.4f}  %>0.5={(scores > 0.5).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
