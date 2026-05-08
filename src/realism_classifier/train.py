"""Training loop, early stopping, and orchestration."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, resolve_device, save_config
from .dataset import build_dataloaders
from .model import RealismMLP, build_model, save_checkpoint


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Tracks a monitored metric and signals when training should stop.

    Args:
        patience:  epochs to wait after last improvement
        min_delta: minimum absolute change to count as improvement
        mode:      "max" (higher is better) or "min" (lower is better)
    """

    def __init__(self, patience: int, min_delta: float = 1e-4, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: Optional[float] = None
        self._counter = 0

    def __call__(self, value: float) -> bool:
        """Return True if training should stop."""
        if self._best is None:
            self._best = value
            return False

        if self.mode == "max":
            improved = value > self._best + self.min_delta
        else:
            improved = value < self._best - self.min_delta

        if improved:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience

    @property
    def best_value(self) -> Optional[float]:
        return self._best

    def state_dict(self) -> dict:
        return {"best": self._best, "counter": self._counter}

    def load_state_dict(self, state: dict) -> None:
        self._best = state["best"]
        self._counter = state["counter"]


# ---------------------------------------------------------------------------
# Metric accumulator
# ---------------------------------------------------------------------------

class MetricTracker:
    """Accumulates weighted running sums; computes epoch-level means."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, key: str, value: float, count: int = 1) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + value * count
        self._counts[key] = self._counts.get(key, 0) + count

    def compute(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Optimizer and scheduler factories
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, config_training) -> torch.optim.Optimizer:
    """Return an Adam or AdamW optimizer."""
    name = config_training.optimizer.lower()
    kwargs = dict(
        lr=config_training.learning_rate,
        weight_decay=config_training.weight_decay,
    )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Use 'adam' or 'adamw'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config_training,
    steps_per_epoch: int,
):
    """
    Return a LR scheduler.

    If warmup_epochs > 0, a LambdaLR warmup is applied for the first
    warmup_epochs * steps_per_epoch steps; the main scheduler then takes over.
    Returns None when scheduler type is "none".
    """
    sched_cfg = config_training.scheduler
    warmup_steps = sched_cfg.warmup_epochs * steps_per_epoch

    # Build main scheduler
    stype = sched_cfg.type.lower()
    if stype == "none":
        main_sched = None
    elif stype == "cosine":
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sched_cfg.T_max
        )
    elif stype == "step":
        main_sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma
        )
    elif stype == "plateau":
        main_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.gamma,
            patience=sched_cfg.patience,
        )
    else:
        raise ValueError(f"Unknown scheduler type '{stype}'.")

    if warmup_steps == 0:
        return main_sched

    # Wrap with linear warmup
    def warmup_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    if main_sched is None:
        return warmup

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main_sched],
        milestones=[warmup_steps],
    )


# ---------------------------------------------------------------------------
# Epoch-level helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: RealismMLP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scheduler=None,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Returns:
        {"loss": float, "accuracy": float}
    """
    model.train()
    tracker = MetricTracker()
    n_correct = 0
    n_total = 0

    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

        bs = embeddings.size(0)
        tracker.update("loss", loss.item(), bs)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        n_correct += (preds == labels.long()).sum().item()
        n_total += bs

    metrics = tracker.compute()
    metrics["accuracy"] = n_correct / n_total if n_total > 0 else 0.0
    return metrics


def validate_one_epoch(
    model: RealismMLP,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """
    Run validation.

    Returns:
        {"loss": float, "accuracy": float, "auc": float, "f1": float}
    """
    from sklearn.metrics import f1_score, roc_auc_score

    model.eval()
    tracker = MetricTracker()
    all_probs: list = []
    all_labels: list = []
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model(embeddings)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            bs = embeddings.size(0)
            tracker.update("loss", loss.item(), bs)
            preds = (probs >= 0.5).long()
            n_correct += (preds == labels.long()).sum().item()
            n_total += bs

            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = tracker.compute()
    metrics["accuracy"] = n_correct / n_total if n_total > 0 else 0.0

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels).astype(int)

    # AUC-ROC (requires both classes present)
    if len(np.unique(labels_arr)) > 1:
        metrics["auc"] = float(roc_auc_score(labels_arr, probs_arr))
    else:
        metrics["auc"] = 0.5

    # F1 at threshold 0.5
    preds_arr = (probs_arr >= 0.5).astype(int)
    metrics["f1"] = float(f1_score(labels_arr, preds_arr, zero_division=0))

    return metrics


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def train(
    config: Config,
    run_dir: Optional[str] = None,
) -> str:
    """
    Full training pipeline.

    1. Seeds, device resolution
    2. DataLoaders, model, optimizer, scheduler, criterion
    3. Epoch loop: train -> validate -> scheduler step -> early stop -> checkpoint
    4. Logs per-epoch metrics to {run_dir}/metrics.csv
    5. Saves run config to {run_dir}/config.yaml
    6. Returns path to best checkpoint

    Args:
        config:  fully populated Config object
        run_dir: output directory for this run; auto-generated if None
    """
    # ---- setup ----
    _seed_everything(config.seed)
    device = resolve_device(config.device)

    if run_dir is None:
        run_dir = _make_run_dir(config.checkpointing.output_dir)
    else:
        Path(run_dir).mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    save_config(config, str(Path(run_dir) / "config.yaml"))

    # ---- data ----
    loaders = build_dataloaders(config.data, config.training, splits=("train", "val"))
    if "train" not in loaders:
        raise RuntimeError("No 'train' split found in data.")
    if "val" not in loaders:
        raise RuntimeError("No 'val' split found in data.")

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    embedding_dim = train_loader.dataset.embedding_dim  # type: ignore[attr-defined]

    # ---- model ----
    model = build_model(config.model, input_dim=embedding_dim).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # ---- criterion ----
    pos_weight = None
    if config.training.pos_weight is not None:
        pos_weight = torch.tensor([config.training.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- optimizer / scheduler ----
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training, steps_per_epoch=len(train_loader))

    # ---- early stopping ----
    es_cfg = config.training.early_stopping
    early_stopper: Optional[EarlyStopping] = None
    if es_cfg.enabled:
        early_stopper = EarlyStopping(
            patience=es_cfg.patience,
            min_delta=es_cfg.min_delta,
            mode=es_cfg.mode,
        )

    # ---- metric logger ----
    metric_path = Path(run_dir) / "metrics.csv"
    best_ckpt_path = str(Path(run_dir) / "best_model.pt")
    ckpt_metric = config.checkpointing.checkpoint_metric  # e.g. "val_auc"
    best_metric_value = float("-inf")

    fieldnames: Optional[list] = None

    # ---- epoch loop ----
    for epoch in range(1, config.training.max_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        # Step epoch-level schedulers
        if scheduler is not None and not isinstance(
            scheduler, (
                torch.optim.lr_scheduler.LambdaLR,
                torch.optim.lr_scheduler.SequentialLR,
            )
        ):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        elapsed = time.time() - t0

        # Flatten metrics for logging
        row = {"epoch": epoch, "elapsed_s": round(elapsed, 2)}
        row.update({f"train_{k}": round(v, 6) for k, v in train_metrics.items()})
        row.update({f"val_{k}": round(v, 6) for k, v in val_metrics.items()})

        # Console output
        print(
            f"Epoch {epoch:04d} | "
            f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"auc={val_metrics.get('auc', 0):.4f} f1={val_metrics.get('f1', 0):.4f} | "
            f"{elapsed:.1f}s"
        )

        # CSV logging
        _append_csv(metric_path, row, fieldnames)
        if fieldnames is None:
            fieldnames = list(row.keys())

        # Checkpointing
        monitor_key = ckpt_metric.replace("val_", "")  # "val_auc" -> "auc"
        current_val = val_metrics.get(monitor_key, val_metrics.get("auc", 0.0))
        if current_val > best_metric_value:
            best_metric_value = current_val
            save_checkpoint(model, optimizer, scheduler, epoch, row, config, best_ckpt_path)

        # Early stopping
        if early_stopper is not None:
            es_metric_key = es_cfg.metric.replace("val_", "")
            es_value = val_metrics.get(es_metric_key, val_metrics.get("loss", 0.0))
            if early_stopper(es_value):
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best {es_cfg.metric}={early_stopper.best_value:.6f})"
                )
                break

    print(f"Training complete. Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_run_dir(base_dir: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def _append_csv(path: Path, row: dict, fieldnames: Optional[list]) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
