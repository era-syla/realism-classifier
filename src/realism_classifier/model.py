"""MLP classifier for DINO embeddings and checkpoint utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RealismMLP(nn.Module):
    """
    Multi-layer perceptron binary classifier on top of DINO embeddings.

    Architecture per hidden layer:
        Linear -> (BatchNorm1d) -> Activation -> Dropout
    Final layer:
        Linear(last_hidden_dim, 1)  -> raw logit

    Use BCEWithLogitsLoss during training (more numerically stable than
    sigmoid + BCELoss).  Use predict_proba() for inference scores.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dims = list(hidden_dims)
        self._dropout = dropout
        self._activation = activation
        self._use_batch_norm = use_batch_norm

        act_fn = _get_activation(activation)
        dims = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_d))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) float tensor
        Returns:
            logits: (B,) float tensor — raw, pre-sigmoid
        """
        return self.net(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return sigmoid probabilities in [0, 1].

        Args:
            x: (B, input_dim) float tensor
        Returns:
            probs: (B,) float tensor
        """
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def model_kwargs(self) -> dict:
        """Dict of constructor kwargs needed to reconstruct this model."""
        return {
            "input_dim": self._input_dim,
            "hidden_dims": self._hidden_dims,
            "dropout": self._dropout,
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config_model, input_dim: int) -> RealismMLP:
    """Instantiate RealismMLP from a ModelConfig and an input dimension."""
    return RealismMLP(
        input_dim=input_dim,
        hidden_dims=config_model.hidden_dims,
        dropout=config_model.dropout,
        activation=config_model.activation,
        use_batch_norm=config_model.use_batch_norm,
    )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: RealismMLP,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    config,
    path: str,
) -> None:
    """
    Save a full training checkpoint.

    The checkpoint includes model_kwargs so the model can be reconstructed
    from the file alone — no separate config file is needed at inference time.
    """
    from dataclasses import asdict

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "model_kwargs": model.model_kwargs,
    }
    try:
        ckpt["config"] = asdict(config)
    except Exception:
        pass
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    """
    Load a raw checkpoint dict from disk without reconstructing the model.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(str(p), map_location=device, weights_only=False)


def load_model_from_checkpoint(path: str, device: str = "cpu") -> RealismMLP:
    """
    Load a checkpoint and reconstruct the RealismMLP, return in eval mode.
    """
    ckpt = load_checkpoint(path, device=device)
    model = RealismMLP(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_activation(name: str):
    """Return an activation class (not instance) by name."""
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(activations)}"
        )
    return activations[name]
