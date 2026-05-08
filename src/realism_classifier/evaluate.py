"""Metrics computation, per-dataset reporting, and evaluation plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def collect_predictions(
    model,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model in eval mode and collect outputs.

    Returns:
        probs:  (N,) float32 sigmoid probabilities
        preds:  (N,) int   hard predictions (0 or 1) at given threshold
        labels: (N,) int   ground-truth labels
    """
    model.eval()
    all_probs: list = []
    all_labels: list = []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            logits = model(embeddings)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    probs_arr = np.array(all_probs, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=int)
    preds_arr = (probs_arr >= threshold).astype(int)
    return probs_arr, preds_arr, labels_arr


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Returns:
        accuracy, precision, recall, f1, auc_roc, avg_precision, threshold
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    preds = (probs >= threshold).astype(int)
    result: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": threshold,
    }
    if len(np.unique(labels)) > 1:
        result["auc_roc"] = float(roc_auc_score(labels, probs))
        result["avg_precision"] = float(average_precision_score(labels, probs))
    else:
        result["auc_roc"] = 0.5
        result["avg_precision"] = float(np.mean(labels))
    return result


def per_dataset_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    dataset_names: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute compute_metrics for each unique dataset name.

    Returns a DataFrame indexed by dataset name, one metric per column.
    """
    records = []
    for ds in np.unique(dataset_names):
        mask = dataset_names == ds
        if mask.sum() == 0:
            continue
        m = compute_metrics(labels[mask], probs[mask], threshold=threshold)
        m["dataset"] = ds
        m["n_samples"] = int(mask.sum())
        records.append(m)
    return pd.DataFrame(records).set_index("dataset")


def score_distribution_stats(
    probs: np.ndarray,
    dataset_names: np.ndarray,
) -> pd.DataFrame:
    """
    Per-dataset summary statistics for realism scores.

    Columns: mean, median, std, min, max, p25, p75, count
    """
    records = []
    for ds in np.unique(dataset_names):
        mask = dataset_names == ds
        scores = probs[mask]
        records.append(
            {
                "dataset": ds,
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "p25": float(np.percentile(scores, 25)),
                "p75": float(np.percentile(scores, 75)),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(records).set_index("dataset")


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    config,
    checkpoint_path: Optional[str] = None,
    split: str = "test",
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Full evaluation pipeline on a labeled split.

    Returns dict with keys:
        global_metrics, per_dataset_metrics, score_distribution, probs, labels
    """
    from .config import resolve_device
    from .dataset import build_dataloaders
    from .model import load_model_from_checkpoint

    device = resolve_device(config.device)
    ckpt_path = checkpoint_path or config.inference.checkpoint_path
    if not ckpt_path:
        raise ValueError("checkpoint_path must be provided or set in config.inference.checkpoint_path")

    loaders = build_dataloaders(config.data, config.training, splits=(split,))
    if split not in loaders:
        raise RuntimeError(f"Split '{split}' not found in data.")

    loader = loaders[split]
    model = load_model_from_checkpoint(ckpt_path, device=device)

    probs, preds, labels = collect_predictions(
        model, loader, device, threshold=config.evaluation.threshold
    )

    # Collect dataset names from metadata
    meta = loader.dataset.get_metadata()  # type: ignore[attr-defined]
    dataset_names = meta["dataset"].values if "dataset" in meta.columns else np.full(len(probs), "unknown")

    global_m = compute_metrics(labels, probs, threshold=config.evaluation.threshold)
    per_ds_m = per_dataset_metrics(labels, probs, dataset_names, threshold=config.evaluation.threshold)
    dist_stats = score_distribution_stats(probs, dataset_names)

    results = {
        "global_metrics": global_m,
        "per_dataset_metrics": per_ds_m,
        "score_distribution": dist_stats,
        "probs": probs,
        "labels": labels,
    }

    if output_dir is not None:
        _save_evaluation_results(results, output_dir, config, probs, labels, dataset_names)

    return results


def _save_evaluation_results(results, output_dir, config, probs, labels, dataset_names):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "global_metrics": results["global_metrics"],
        "per_dataset_metrics": results["per_dataset_metrics"].reset_index().to_dict(orient="records"),
    }
    with open(out / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # CSVs
    results["per_dataset_metrics"].reset_index().to_csv(out / "per_dataset_metrics.csv", index=False)
    results["score_distribution"].reset_index().to_csv(out / "score_distribution.csv", index=False)

    # Plots
    try:
        plot_roc_curve(labels, probs, str(out / "roc_curve.png"))
        plot_score_distributions(
            probs,
            labels,
            dataset_names,
            str(out),
            plot_format=config.comparison.plot_format,
        )
    except Exception as exc:
        print(f"Warning: could not generate plots: {exc}")

    print(f"Evaluation results saved to {output_dir}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: str,
) -> None:
    """Plot and save ROC curve with AUC annotation."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_score_distributions(
    probs: np.ndarray,
    labels: np.ndarray,
    dataset_names: np.ndarray,
    output_dir: str,
    plot_format: str = "png",
) -> None:
    """
    One subplot per dataset: KDE/histogram of realism scores color-coded by label.
    """
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ds in np.unique(dataset_names):
        mask = dataset_names == ds
        scores = probs[mask]
        ds_labels = labels[mask]

        fig, ax = plt.subplots(figsize=(6, 4))
        for lbl, lname, color in [(1, "Realistic", "steelblue"), (0, "Synthetic", "tomato")]:
            sub = scores[ds_labels == lbl]
            if len(sub) > 0:
                ax.hist(sub, bins=30, alpha=0.6, label=f"{lname} (n={len(sub)})", color=color, density=True)
        ax.set_xlabel("Realism Score")
        ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution — {ds}")
        ax.legend()
        fig.tight_layout()
        fname = f"score_dist_{ds.replace('/', '_')}.{plot_format}"
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)
