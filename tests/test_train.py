"""Integration test: short training run end-to-end."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from realism_classifier.config import Config, DataConfig, TrainingConfig, EarlyStoppingConfig
from realism_classifier.train import EarlyStopping, MetricTracker, train


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

def test_early_stopping_max_mode():
    es = EarlyStopping(patience=3, min_delta=0.01, mode="max")
    assert not es(0.5)
    assert not es(0.6)   # improvement
    assert not es(0.6)   # no improvement, counter=1
    assert not es(0.6)   # counter=2
    assert es(0.6)       # counter=3 -> stop


def test_early_stopping_min_mode():
    es = EarlyStopping(patience=2, min_delta=0.0, mode="min")
    assert not es(1.0)
    assert not es(0.5)   # improvement
    assert not es(0.5)   # no improvement, counter=1
    assert es(0.5)       # counter=2 -> stop


def test_metric_tracker():
    t = MetricTracker()
    t.update("loss", 2.0, count=10)
    t.update("loss", 1.0, count=10)
    result = t.compute()
    assert abs(result["loss"] - 1.5) < 1e-6
    t.reset()
    assert t.compute() == {}


# ---------------------------------------------------------------------------
# Integration: short training run
# ---------------------------------------------------------------------------

def make_synthetic_npz(tmp_dir, n_train=80, n_val=20, d=64):
    """Create a minimal .npz + metadata.csv for a training run."""
    train_emb = np.random.randn(n_train, d).astype(np.float32)
    val_emb = np.random.randn(n_val, d).astype(np.float32)

    emb_path = os.path.join(tmp_dir, "embeddings.npz")
    np.savez(emb_path, train=train_emb, val=val_emb)

    train_meta = pd.DataFrame({
        "sample_id": [f"tr_{i}" for i in range(n_train)],
        "label": np.random.randint(0, 2, n_train),
        "split": "train",
        "dataset": "synth_train",
    })
    val_meta = pd.DataFrame({
        "sample_id": [f"vl_{i}" for i in range(n_val)],
        "label": np.random.randint(0, 2, n_val),
        "split": "val",
        "dataset": "synth_val",
    })
    meta = pd.concat([train_meta, val_meta], ignore_index=True)
    meta_path = os.path.join(tmp_dir, "metadata.csv")
    meta.to_csv(meta_path, index=False)
    return emb_path, meta_path


def make_config(emb_path, meta_path, run_dir, d=64):
    from realism_classifier.config import (
        CheckpointingConfig, EarlyStoppingConfig, EvaluationConfig,
        InferenceConfig, ComparisonConfig, ModelConfig, SchedulerConfig,
        TrainingConfig,
    )
    config = Config(
        data=DataConfig(
            embeddings_path=emb_path,
            metadata_csv=meta_path,
            embedding_dim=d,
            num_workers=0,
        ),
        model=ModelConfig(hidden_dims=[32, 16], dropout=0.0, use_batch_norm=False),
        training=TrainingConfig(
            batch_size=16,
            max_epochs=3,
            learning_rate=1e-3,
            scheduler=SchedulerConfig(type="none", warmup_epochs=0),
            early_stopping=EarlyStoppingConfig(enabled=False),
        ),
        checkpointing=CheckpointingConfig(output_dir=run_dir),
        evaluation=EvaluationConfig(),
        inference=InferenceConfig(),
        comparison=ComparisonConfig(),
        seed=0,
        device="cpu",
    )
    return config


def test_train_short_run():
    with tempfile.TemporaryDirectory() as tmp:
        emb_path, meta_path = make_synthetic_npz(tmp)
        run_dir = os.path.join(tmp, "runs")
        config = make_config(emb_path, meta_path, run_dir)

        best_ckpt = train(config, run_dir=run_dir)
        assert os.path.exists(best_ckpt), f"Checkpoint not found: {best_ckpt}"

        # Metrics CSV should exist
        metrics_csv = os.path.join(run_dir, "metrics.csv")
        assert os.path.exists(metrics_csv)
        df = pd.read_csv(metrics_csv)
        assert len(df) == 3  # 3 epochs
        assert "train_loss" in df.columns
        assert "val_auc" in df.columns


def test_train_checkpoint_loadable():
    """Checkpoint saved during training must be loadable."""
    from realism_classifier.model import load_model_from_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        emb_path, meta_path = make_synthetic_npz(tmp)
        run_dir = os.path.join(tmp, "runs")
        config = make_config(emb_path, meta_path, run_dir)
        best_ckpt = train(config, run_dir=run_dir)

        model = load_model_from_checkpoint(best_ckpt, device="cpu")
        import torch
        x = torch.randn(4, 64)
        probs = model.predict_proba(x)
        assert probs.shape == (4,)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0
