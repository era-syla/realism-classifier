"""Tests for dataset loading and DataLoader factory."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Allow imports without installing the package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from realism_classifier.dataset import (
    EmbeddingDataset,
    align_embeddings_and_metadata,
    build_inference_dataloader,
    load_embeddings,
    load_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_embeddings(n=50, d=64) -> np.ndarray:
    return np.random.randn(n, d).astype(np.float32)


def make_metadata(n=50, split="train") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i:04d}" for i in range(n)],
            "label": np.random.randint(0, 2, size=n),
            "split": split,
            "dataset": "test_ds",
        }
    )


# ---------------------------------------------------------------------------
# EmbeddingDataset
# ---------------------------------------------------------------------------

def test_embedding_dataset_with_labels():
    emb = make_embeddings(10, 32)
    meta = make_metadata(10)
    ds = EmbeddingDataset(emb, meta, has_labels=True)
    assert len(ds) == 10
    assert ds.embedding_dim == 32
    x, y = ds[0]
    assert x.shape == (32,)
    assert y.item() in (0.0, 1.0)


def test_embedding_dataset_without_labels():
    emb = make_embeddings(8, 16)
    meta = make_metadata(8)
    ds = EmbeddingDataset(emb, meta, has_labels=False)
    x, sid = ds[0]
    assert x.shape == (16,)
    assert isinstance(sid, str)


def test_embedding_dataset_length_mismatch():
    emb = make_embeddings(10)
    meta = make_metadata(9)
    with pytest.raises(ValueError, match="same length"):
        EmbeddingDataset(emb, meta, has_labels=True)


def test_embedding_dataset_missing_label_column():
    emb = make_embeddings(5)
    meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(5)]})
    with pytest.raises(ValueError, match="label"):
        EmbeddingDataset(emb, meta, has_labels=True)


# ---------------------------------------------------------------------------
# load_embeddings
# ---------------------------------------------------------------------------

def test_load_embeddings_npy():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "emb.npy")
        arr = make_embeddings(20, 64)
        np.save(path, arr)
        result = load_embeddings(path)
        assert "all" in result
        assert result["all"].shape == (20, 64)
        assert result["all"].dtype == np.float32


def test_load_embeddings_npz():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "emb.npz")
        train_emb = make_embeddings(30, 64)
        val_emb = make_embeddings(10, 64)
        np.savez(path, train=train_emb, val=val_emb)
        result = load_embeddings(path)
        assert "train" in result
        assert "val" in result
        assert result["train"].shape == (30, 64)


def test_load_embeddings_missing_file():
    with pytest.raises(FileNotFoundError):
        load_embeddings("/nonexistent/path.npy")


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------

def test_load_metadata_valid():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "meta.csv")
        make_metadata(10).to_csv(path, index=False)
        df = load_metadata(path)
        assert len(df) == 10
        assert "sample_id" in df.columns


def test_load_metadata_missing_column():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "meta.csv")
        pd.DataFrame({"sample_id": ["a", "b"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_metadata(path)


# ---------------------------------------------------------------------------
# align_embeddings_and_metadata
# ---------------------------------------------------------------------------

def test_align_length_mismatch():
    emb = make_embeddings(10)
    meta = make_metadata(11)
    with pytest.raises(ValueError, match="Length mismatch"):
        align_embeddings_and_metadata(emb, meta)


# ---------------------------------------------------------------------------
# build_inference_dataloader
# ---------------------------------------------------------------------------

def test_build_inference_dataloader_no_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "emb.npy")
        np.save(path, make_embeddings(12, 32))
        loader, meta = build_inference_dataloader(path, metadata_csv=None, batch_size=4, num_workers=0)
        assert meta is not None
        assert len(meta) == 12
        # One batch
        batch = next(iter(loader))
        assert batch[0].shape[1] == 32
