"""Tests for inference and comparison pipelines."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from realism_classifier.model import RealismMLP, save_checkpoint
from realism_classifier.inference import score_dataset
from realism_classifier.compare import (
    compute_dataset_summary,
    load_scored_csvs,
    rank_datasets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_dummy_checkpoint(tmp_dir, input_dim=32):
    model = RealismMLP(input_dim=input_dim, hidden_dims=[16], dropout=0.0, use_batch_norm=False)
    ckpt_path = os.path.join(tmp_dir, "model.pt")
    opt = torch.optim.Adam(model.parameters())
    save_checkpoint(model, opt, None, epoch=1, metrics={}, config=None, path=ckpt_path)
    return ckpt_path


def make_inference_data(tmp_dir, n=20, d=32, with_metadata=True):
    emb = np.random.randn(n, d).astype(np.float32)
    emb_path = os.path.join(tmp_dir, "emb.npy")
    np.save(emb_path, emb)

    meta_path = None
    if with_metadata:
        meta = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "dataset": "ds_a",
        })
        meta_path = os.path.join(tmp_dir, "meta.csv")
        meta.to_csv(meta_path, index=False)

    return emb_path, meta_path


def make_config(tmp_dir, ckpt_path, d=32):
    from realism_classifier.config import (
        Config, DataConfig, ModelConfig, TrainingConfig, CheckpointingConfig,
        EvaluationConfig, InferenceConfig, ComparisonConfig,
        SchedulerConfig, EarlyStoppingConfig,
    )
    return Config(
        data=DataConfig(embeddings_path="", metadata_csv="", embedding_dim=d, num_workers=0),
        model=ModelConfig(hidden_dims=[16], dropout=0.0, use_batch_norm=False),
        training=TrainingConfig(
            scheduler=SchedulerConfig(type="none"),
            early_stopping=EarlyStoppingConfig(enabled=False),
        ),
        checkpointing=CheckpointingConfig(output_dir=tmp_dir),
        evaluation=EvaluationConfig(),
        inference=InferenceConfig(
            checkpoint_path=ckpt_path,
            output_csv=os.path.join(tmp_dir, "scores.csv"),
            batch_size=8,
        ),
        comparison=ComparisonConfig(
            scores_dir=os.path.join(tmp_dir, "scores"),
            output_dir=os.path.join(tmp_dir, "comparison"),
        ),
        seed=0,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------

def test_score_dataset_with_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = save_dummy_checkpoint(tmp, input_dim=32)
        emb_path, meta_path = make_inference_data(tmp, n=20, d=32, with_metadata=True)
        config = make_config(tmp, ckpt_path)

        out_df = score_dataset(
            config,
            embeddings_path=emb_path,
            metadata_csv=meta_path,
        )
        assert len(out_df) == 20
        assert "realism_score" in out_df.columns
        assert out_df["realism_score"].between(0.0, 1.0).all()
        assert "sample_id" in out_df.columns


def test_score_dataset_without_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = save_dummy_checkpoint(tmp, input_dim=32)
        emb_path, _ = make_inference_data(tmp, n=10, d=32, with_metadata=False)
        config = make_config(tmp, ckpt_path)

        out_df = score_dataset(config, embeddings_path=emb_path, metadata_csv=None)
        assert len(out_df) == 10
        assert "realism_score" in out_df.columns


def test_score_dataset_output_csv_written():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = save_dummy_checkpoint(tmp, input_dim=32)
        emb_path, meta_path = make_inference_data(tmp, n=5, d=32)
        out_csv = os.path.join(tmp, "out_scores.csv")
        config = make_config(tmp, ckpt_path)

        score_dataset(config, embeddings_path=emb_path, metadata_csv=meta_path, output_csv=out_csv)
        assert os.path.exists(out_csv)
        df = pd.read_csv(out_csv)
        assert "realism_score" in df.columns


# ---------------------------------------------------------------------------
# Compare tests
# ---------------------------------------------------------------------------

def test_load_scored_csvs():
    with tempfile.TemporaryDirectory() as tmp:
        scores_dir = os.path.join(tmp, "scores")
        os.makedirs(scores_dir)
        for name in ["ds_a", "ds_b"]:
            pd.DataFrame({
                "sample_id": [f"s{i}" for i in range(5)],
                "dataset": name,
                "realism_score": np.random.rand(5),
            }).to_csv(os.path.join(scores_dir, f"{name}.csv"), index=False)

        df = load_scored_csvs(scores_dir)
        assert len(df) == 10
        assert set(df["dataset"].unique()) == {"ds_a", "ds_b"}


def test_compute_dataset_summary_and_rank():
    df = pd.DataFrame({
        "dataset": ["A"] * 10 + ["B"] * 10,
        "realism_score": [0.8] * 10 + [0.3] * 10,
    })
    summary = compute_dataset_summary(df)
    assert "A" in summary.index
    assert summary.loc["A", "mean"] > summary.loc["B", "mean"]

    ranked = rank_datasets(summary)
    assert ranked.iloc[0]["rank"] == 1
    assert ranked.index[0] == "A"
