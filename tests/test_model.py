"""Tests for RealismMLP and checkpoint utilities."""

import os
import tempfile

import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from realism_classifier.model import (
    RealismMLP,
    load_model_from_checkpoint,
    save_checkpoint,
)


def make_model(input_dim=64) -> RealismMLP:
    return RealismMLP(
        input_dim=input_dim,
        hidden_dims=[32, 16],
        dropout=0.1,
        activation="relu",
        use_batch_norm=True,
    )


def test_forward_shape():
    model = make_model(64)
    x = torch.randn(8, 64)
    out = model(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_predict_proba_range():
    model = make_model(64)
    x = torch.randn(16, 64)
    probs = model.predict_proba(x)
    assert probs.shape == (16,)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0


def test_count_parameters():
    model = make_model(64)
    assert model.count_parameters() > 0


def test_activation_gelu():
    model = RealismMLP(input_dim=32, hidden_dims=[16], activation="gelu", use_batch_norm=False)
    out = model(torch.randn(4, 32))
    assert out.shape == (4,)


def test_activation_leaky_relu():
    model = RealismMLP(input_dim=32, hidden_dims=[16], activation="leaky_relu", use_batch_norm=False)
    out = model(torch.randn(4, 32))
    assert out.shape == (4,)


def test_invalid_activation():
    with pytest.raises(ValueError, match="Unknown activation"):
        RealismMLP(input_dim=32, hidden_dims=[16], activation="swish")


def test_model_kwargs_roundtrip():
    model = make_model(64)
    kwargs = model.model_kwargs
    model2 = RealismMLP(**kwargs)
    # Same architecture -> same number of parameters
    assert model.count_parameters() == model2.count_parameters()


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "ckpt.pt")
        model = make_model(64)
        optimizer = torch.optim.Adam(model.parameters())
        save_checkpoint(model, optimizer, None, epoch=1, metrics={"val_auc": 0.9}, config=None, path=ckpt_path)

        loaded = load_model_from_checkpoint(ckpt_path, device="cpu")
        assert loaded.count_parameters() == model.count_parameters()

        # Same output for same input
        x = torch.randn(4, 64)
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = loaded(x)
        assert torch.allclose(out1, out2), "Checkpoint roundtrip produced different outputs"
