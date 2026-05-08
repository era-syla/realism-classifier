"""Score new embedding files with a trained model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def run_inference(
    model,
    dataloader: DataLoader,
    device: str,
) -> np.ndarray:
    """
    Run model.predict_proba over an unlabeled DataLoader.

    Returns:
        probs: (N,) float32 realism scores in [0, 1]
    """
    model.eval()
    all_probs: list = []

    with torch.no_grad():
        for batch in dataloader:
            # Inference DataLoader yields (embedding, sample_id)
            embeddings = batch[0].to(device)
            probs = torch.sigmoid(model(embeddings)).cpu().numpy()
            all_probs.extend(probs.tolist())

    return np.array(all_probs, dtype=np.float32)


def score_dataset(
    config,
    embeddings_path: Optional[str] = None,
    metadata_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full inference pipeline for a single dataset.

    Explicit arguments take precedence over values in config.inference.

    Steps:
    1. Resolve paths (explicit args > config)
    2. Load model from checkpoint
    3. Build inference DataLoader (no labels)
    4. Run inference
    5. Merge scores with metadata
    6. Save output CSV
    7. Return output DataFrame

    Output DataFrame columns:
        sample_id, realism_score, [dataset, object_id, view, + any extra metadata columns]
    """
    from .config import resolve_device
    from .dataset import build_inference_dataloader
    from .model import load_model_from_checkpoint

    # Resolve paths
    emb_path = embeddings_path or config.inference.embeddings_path
    meta_path = metadata_csv or config.inference.metadata_csv
    out_csv = output_csv or config.inference.output_csv
    ckpt_path = checkpoint_path or config.inference.checkpoint_path

    if not emb_path:
        raise ValueError("embeddings_path must be provided or set in config.inference.embeddings_path")
    if not ckpt_path:
        raise ValueError("checkpoint_path must be provided or set in config.inference.checkpoint_path")

    device = resolve_device(config.device)
    model = load_model_from_checkpoint(ckpt_path, device=device)

    loader, meta = build_inference_dataloader(
        embeddings_path=emb_path,
        metadata_csv=meta_path,
        batch_size=config.inference.batch_size,
        num_workers=config.data.num_workers,
    )

    probs = run_inference(model, loader, device)

    # Build output DataFrame — preserve all metadata columns
    if meta is not None:
        out_df = meta.copy().reset_index(drop=True)
    else:
        out_df = pd.DataFrame({"sample_id": [f"sample_{i:06d}" for i in range(len(probs))]})

    out_df["realism_score"] = probs

    # Reorder so sample_id and realism_score come first
    front_cols = [c for c in ["sample_id", "realism_score"] if c in out_df.columns]
    other_cols = [c for c in out_df.columns if c not in front_cols]
    out_df = out_df[front_cols + other_cols]

    # Save
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        print(f"Scores saved to {out_csv}  ({len(out_df)} samples)")

    return out_df
