"""
Standalone inference script for the realism classifier.

Usage:
    python scripts/run_inference.py \
        --pkl /path/to/embeddings.pkl \
        --checkpoint outputs/runs/run_03/best_model.pt \
        --output outputs/scores.csv

The pkl file must contain a dict with keys:
    'embeddings': np.ndarray of shape (N, 768) float32
    'paths':      list of N file paths (optional)
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


def load_pkl(pkl_path):
    import pickle
    with open(pkl_path, "rb") as f:
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


def run_inference(emb, model, device, batch_size=512):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(emb), batch_size):
            b = torch.from_numpy(emb[i:i + batch_size]).to(device)
            s = torch.sigmoid(model(b).squeeze(-1)).cpu().numpy()
            scores.append(s)
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser(description="Run realism classifier on a pkl file")
    parser.add_argument("--pkl",        required=True,  help="Path to .pkl file with embeddings")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pt model checkpoint")
    parser.add_argument("--output",     required=True,  help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device",     default="auto", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    print(f"Loading embeddings from {args.pkl} ...")
    emb, paths = load_pkl(args.pkl)
    print(f"  {len(emb):,} samples, dim={emb.shape[1]}")

    device = resolve_device(args.device)
    print(f"Loading model from {args.checkpoint} (device={device}) ...")
    model = load_model_from_checkpoint(args.checkpoint, device=device)

    print("Running inference ...")
    scores = run_inference(emb, model, device, batch_size=args.batch_size)

    filenames = [os.path.basename(p) for p in paths]
    df = pd.DataFrame({"filename": filenames, "path": paths, "realism_score": scores})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nResults saved to {args.output}")
    print(f"  n={len(scores):,}  mean={scores.mean():.4f}  median={float(np.median(scores)):.4f}  %>0.5={(scores>0.5).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
