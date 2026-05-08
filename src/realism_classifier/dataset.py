"""Dataset classes and DataLoader factories for DINO embedding data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Required columns for each mode
# ---------------------------------------------------------------------------

_TRAIN_REQUIRED = {"sample_id", "label", "split", "dataset"}
_INFER_REQUIRED = {"sample_id"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for precomputed DINO embedding vectors.

    Args:
        embeddings: float32 array of shape (N, D)
        metadata:   DataFrame with at least 'sample_id'; 'label' required when
                    has_labels=True.  Must be positionally aligned to embeddings.
        has_labels: if False, __getitem__ returns (embedding, sample_id).
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        has_labels: bool = True,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (N, D), got shape {embeddings.shape}"
            )
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                "must have the same length"
            )
        if has_labels and "label" not in metadata.columns:
            raise ValueError("metadata must contain a 'label' column when has_labels=True")

        self._embeddings = embeddings.astype(np.float32)
        self._metadata = metadata.reset_index(drop=True)
        self._has_labels = has_labels

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._embeddings)

    def __getitem__(self, idx: int) -> Tuple:
        emb = torch.from_numpy(self._embeddings[idx])
        if self._has_labels:
            label = torch.tensor(
                float(self._metadata.iloc[idx]["label"]), dtype=torch.float32
            )
            return emb, label
        else:
            sample_id = str(self._metadata.iloc[idx]["sample_id"])
            return emb, sample_id

    @property
    def embedding_dim(self) -> int:
        return self._embeddings.shape[1]

    def get_metadata(self) -> pd.DataFrame:
        """Return a copy of the aligned metadata DataFrame."""
        return self._metadata.copy()


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from a .npy or .npz file.

    Returns:
        For .npy:  {"all": array of shape (N, D)}
        For .npz:  dict keyed by split name, e.g. {"train": ..., "val": ..., "test": ...}
    All arrays are cast to float32.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    if p.suffix == ".npy":
        arr = np.load(str(p)).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return {"all": arr}
    elif p.suffix == ".npz":
        npz = np.load(str(p))
        return {k: npz[k].astype(np.float32) for k in npz.files}
    else:
        raise ValueError(f"Unsupported embeddings format: {p.suffix}. Use .npy or .npz")


def load_metadata(path: str, required_columns: Optional[set] = None) -> pd.DataFrame:
    """
    Load metadata CSV and validate required columns.

    Args:
        path:             path to CSV file
        required_columns: set of column names that must be present;
                          defaults to _TRAIN_REQUIRED
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    df = pd.read_csv(str(p))
    if required_columns is None:
        required_columns = _TRAIN_REQUIRED

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata CSV is missing required columns: {sorted(missing)}"
        )
    return df


def align_embeddings_and_metadata(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Validate positional alignment between embeddings and metadata rows.

    Raises ValueError if lengths differ.
    Returns (embeddings, metadata) with metadata index reset.
    """
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Length mismatch: embeddings has {len(embeddings)} rows, "
            f"metadata has {len(metadata)} rows. They must be positionally aligned."
        )
    return embeddings, metadata.reset_index(drop=True)


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def build_dataloaders(
    config_data,
    config_training,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders for one or more named splits.

    Supports two data layouts:
    - .npz with split keys: embeddings file has keys matching split names.
    - .npy with 'split' column: single embedding file; metadata CSV has a
      'split' column used to partition rows.

    Returns a dict keyed by split name. Only splits that exist in the data
    are included — missing splits are silently skipped.
    """
    emb_dict = load_embeddings(config_data.embeddings_path)
    meta_all = load_metadata(config_data.metadata_csv, required_columns=_TRAIN_REQUIRED)

    loaders: Dict[str, DataLoader] = {}

    if "all" in emb_dict:
        # Single .npy file — partition by 'split' column in metadata
        embeddings_all, meta_all = align_embeddings_and_metadata(emb_dict["all"], meta_all)
        for split in splits:
            mask = meta_all["split"] == split
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            emb_split = embeddings_all[idx]
            meta_split = meta_all[mask].reset_index(drop=True)
            ds = EmbeddingDataset(emb_split, meta_split, has_labels=True)
            loaders[split] = _make_loader(ds, config_training, shuffle=(split == "train"), num_workers=config_data.num_workers)
    else:
        # .npz file with per-split keys
        for split in splits:
            if split not in emb_dict:
                continue
            meta_split = meta_all[meta_all["split"] == split].reset_index(drop=True)
            emb_split, meta_split = align_embeddings_and_metadata(emb_dict[split], meta_split)
            ds = EmbeddingDataset(emb_split, meta_split, has_labels=True)
            loaders[split] = _make_loader(ds, config_training, shuffle=(split == "train"), num_workers=config_data.num_workers)

    return loaders


def build_inference_dataloader(
    embeddings_path: str,
    metadata_csv: Optional[str],
    batch_size: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[pd.DataFrame]]:
    """
    Build a DataLoader for inference (no labels required).

    Returns:
        (dataloader, metadata_df_or_None)
    """
    emb_dict = load_embeddings(embeddings_path)
    # Flatten: take the first (and typically only) key
    if "all" in emb_dict:
        embeddings = emb_dict["all"]
    else:
        # npz with split keys — concatenate all
        keys = list(emb_dict.keys())
        embeddings = np.concatenate([emb_dict[k] for k in keys], axis=0)

    if metadata_csv is not None:
        meta = load_metadata(metadata_csv, required_columns=_INFER_REQUIRED)
        embeddings, meta = align_embeddings_and_metadata(embeddings, meta)
    else:
        n = len(embeddings)
        meta = pd.DataFrame({"sample_id": [f"sample_{i:06d}" for i in range(n)]})

    ds = EmbeddingDataset(embeddings, meta, has_labels=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, meta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_loader(
    dataset: EmbeddingDataset,
    config_training,
    shuffle: bool,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config_training.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
