"""
Comprehensive test-set analysis and visualization script.

Usage:
    python scripts/test_analysis.py \
        --config configs/default.yaml \
        --checkpoint outputs/runs/run_02/best_model.pt \
        --output-dir outputs/test_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Dataset name mapping
# ---------------------------------------------------------------------------

DATASET_NAMES = {
    # Training datasets
    "bench1A":          "ABC sketch+extrude",
    "bench1B":          "ABC all ops",
    "fusion360":        "Fusion360",
    "fusion360_steps":  "Fusion360",
    "Ghadi's SynCAD":   "SynCAD",
    "ghadi_0023":       "SynCAD",
    "cad_recode":       "CADRecode",
    "cad_recode_100k":  "CADRecode",
    "17":               "SynCAD-17",
    # Held-out / inference-only datasets
    "era_data_1":       "GemCAD data",
    "cadevolve":        "CADEvolve",
    "synthbal_orig":    "SynthBal original",
    "synthbal_aug":     "SynthBal augmented",
}

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT   = "#0f3460"
COLORS   = {
    "real":  "#4fc3f7",
    "synth": "#ef5350",
}
DS_PALETTE = [
    "#4fc3f7", "#81c784", "#ffb74d", "#ce93d8",
    "#f06292", "#80cbc4", "#ffcc02", "#ff8a65",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remap_datasets(names: np.ndarray) -> np.ndarray:
    return np.array([DATASET_NAMES.get(n, n) for n in names])


def styled_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=DARK_BG)
    return fig


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    if title:  ax.set_title(title, color="white", fontsize=11, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color="#aaaacc", fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color="#aaaacc", fontsize=9)


# ---------------------------------------------------------------------------
# 1. ROC + PR curves
# ---------------------------------------------------------------------------

def plot_roc_pr(labels, probs, out_dir):
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)
    fig.suptitle("ROC & Precision-Recall Curves  (test set)", color="white", fontsize=13, y=1.01)

    ax1.set_facecolor(PANEL_BG)
    ax1.plot(fpr, tpr, color=COLORS["real"], lw=2, label=f"AUC = {auc:.4f}")
    ax1.plot([0,1],[0,1], "--", color="#555577", lw=1)
    ax1.fill_between(fpr, tpr, alpha=0.15, color=COLORS["real"])
    style_ax(ax1, "ROC Curve", "False Positive Rate", "True Positive Rate")
    ax1.legend(facecolor=ACCENT, labelcolor="white", fontsize=10)

    ax2.set_facecolor(PANEL_BG)
    ax2.plot(rec, prec, color=COLORS["synth"], lw=2, label=f"AP = {ap:.4f}")
    ax2.fill_between(rec, prec, alpha=0.15, color=COLORS["synth"])
    style_ax(ax2, "Precision-Recall Curve", "Recall", "Precision")
    ax2.legend(facecolor=ACCENT, labelcolor="white", fontsize=10)

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        for spine in ax.spines.values(): spine.set_edgecolor("#444466")
        ax.tick_params(colors="white")

    fig.tight_layout()
    fig.savefig(out_dir / "01_roc_pr.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  AUC-ROC: {auc:.4f}   AP: {ap:.4f}")
    return auc, ap


# ---------------------------------------------------------------------------
# 2. Score distribution: real vs synthetic (global)
# ---------------------------------------------------------------------------

def plot_score_dist_global(labels, probs, out_dir):
    fig, ax = styled_fig(10, 5), None
    ax = fig.add_subplot(111)
    style_ax(ax, "Score Distribution on Test Set  (Real vs Synthetic)",
             "Realism Score", "Density")
    ax.set_facecolor(PANEL_BG)

    for lbl, name, color in [(1, "Real", COLORS["real"]), (0, "Synthetic", COLORS["synth"])]:
        scores = probs[labels == lbl]
        ax.hist(scores, bins=60, density=True, alpha=0.65,
                color=color, label=f"{name}  (n={len(scores):,})", edgecolor="none")

    ax.axvline(0.5, color="#ffcc02", lw=1.5, linestyle="--", label="threshold = 0.5")
    ax.legend(facecolor=ACCENT, labelcolor="white", fontsize=10)
    for spine in ax.spines.values(): spine.set_edgecolor("#444466")
    ax.tick_params(colors="white")

    fig.tight_layout()
    fig.savefig(out_dir / "02_score_dist_global.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Per-dataset violin plot
# ---------------------------------------------------------------------------

def plot_per_dataset_violin(labels, probs, ds_names, out_dir):
    unique_ds = sorted(set(ds_names))
    real_ds  = [d for d in unique_ds if np.mean(labels[ds_names == d]) > 0.5]
    synth_ds = [d for d in unique_ds if np.mean(labels[ds_names == d]) <= 0.5]
    ordered  = real_ds + synth_ds

    fig, ax = styled_fig(max(10, len(ordered) * 1.4), 6), None
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL_BG)

    data_to_plot = [probs[ds_names == d] for d in ordered]
    label_means  = [np.mean(labels[ds_names == d]) for d in ordered]

    parts = ax.violinplot(data_to_plot, positions=range(len(ordered)),
                          showmedians=True, showextrema=False)

    for i, (pc, lm) in enumerate(zip(parts["bodies"], label_means)):
        color = COLORS["real"] if lm > 0.5 else COLORS["synth"]
        pc.set_facecolor(color); pc.set_alpha(0.7); pc.set_edgecolor("#333355")
    parts["cmedians"].set_color("#ffcc02"); parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=30, ha="right", color="white", fontsize=9)
    ax.axhline(0.5, color="#ffcc02", lw=1, linestyle="--", alpha=0.6)
    ax.axvline(len(real_ds) - 0.5, color="#555577", lw=1.5, linestyle=":")
    ax.text(len(real_ds)/2 - 0.5, 1.02, "Real datasets", color=COLORS["real"],
            ha="center", fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(len(real_ds) + len(synth_ds)/2 - 0.5, 1.02, "Synthetic datasets",
            color=COLORS["synth"], ha="center", fontsize=9,
            transform=ax.get_xaxis_transform())
    style_ax(ax, "Score Distribution per Dataset  (test set)", "", "Realism Score")
    for spine in ax.spines.values(): spine.set_edgecolor("#444466")
    ax.tick_params(colors="white")
    ax.set_ylim(-0.05, 1.08)

    fig.tight_layout()
    fig.savefig(out_dir / "03_per_dataset_violin.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(labels, probs, threshold, out_dir):
    from sklearn.metrics import confusion_matrix

    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    # Row-normalize: each row (true class) sums to 100%
    counts = np.array([[tn, fp], [fn, tp]])
    row_totals = counts.sum(axis=1, keepdims=True)
    pct = counts / row_totals * 100   # % within each true class

    fig, ax = styled_fig(7, 6), None
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)

    # Color: correct diagonal green-ish, off-diagonal red-ish
    cell_colors = np.array([
        [COLORS["real"],  "#b71c1c"],   # TN correct, FP wrong  (synth row)
        ["#b71c1c",  COLORS["synth"]],  # FN wrong,  TP correct (real row)
    ])
    # We still draw imshow for background shading
    im = ax.imshow(pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)

    labels_text = [
        [f"{pct[0,0]:.1f}%\ncorrect\n({tn:,})",  f"{pct[0,1]:.1f}%\nwrong\n({fp:,})"],
        [f"{pct[1,0]:.1f}%\nwrong\n({fn:,})",     f"{pct[1,1]:.1f}%\ncorrect\n({tp:,})"],
    ]
    for i in range(2):
        for j in range(2):
            is_correct = (i == j)
            ax.text(j, i, labels_text[i][j],
                    ha="center", va="center",
                    color="white", fontsize=12,
                    fontweight="bold" if is_correct else "normal")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nSynthetic", "Predicted\nReal"], color="white", fontsize=10)
    ax.set_yticklabels(["True\nSynthetic", "True\nReal"], color="white", fontsize=10)
    ax.set_title(f"Confusion Matrix  (threshold = {threshold})\n% normalized within each true class",
                 color="white", fontsize=12, pad=12)
    for spine in ax.spines.values(): spine.set_edgecolor("#444466")

    total = counts.sum()
    acc = (tp + tn) / total
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ax.text(0.5, -0.08,
            f"Overall Accuracy: {acc*100:.1f}%     TPR (Recall): {tpr*100:.1f}%     TNR (Specificity): {tnr*100:.1f}%",
            ha="center", color="#aaaacc", fontsize=9, transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_dir / "04_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Per-dataset accuracy bar chart
# ---------------------------------------------------------------------------

def plot_per_dataset_accuracy(labels, probs, ds_names, threshold, out_dir):
    from sklearn.metrics import accuracy_score, roc_auc_score

    records = []
    for ds in sorted(set(ds_names)):
        mask = ds_names == ds
        l, p = labels[mask], probs[mask]
        acc = accuracy_score(l, (p >= threshold).astype(int))
        mean_score = p.mean()
        n = mask.sum()
        true_label = "Real" if l.mean() > 0.5 else "Synthetic"
        records.append({"dataset": ds, "accuracy": acc, "mean_score": mean_score,
                        "n": n, "true_label": true_label})

    df = pd.DataFrame(records).sort_values("accuracy", ascending=True)

    fig, ax = styled_fig(10, max(5, len(df) * 0.55)), None
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL_BG)

    colors = [COLORS["real"] if r == "Real" else COLORS["synth"] for r in df["true_label"]]
    bars = ax.barh(df["dataset"], df["accuracy"], color=colors, alpha=0.8, edgecolor="none")

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{row['accuracy']:.3f}  (n={row['n']:,})",
                va="center", color="white", fontsize=8)

    ax.axvline(0.5, color="#ffcc02", lw=1, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1.15)
    style_ax(ax, "Per-Dataset Accuracy  (test set)", "Accuracy", "")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444466")

    from matplotlib.patches import Patch
    legend = [Patch(color=COLORS["real"], label="Real dataset"),
              Patch(color=COLORS["synth"], label="Synthetic dataset")]
    ax.legend(handles=legend, facecolor=ACCENT, labelcolor="white", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "05_per_dataset_accuracy.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# 6. Dataset composition pie charts (test set)
# ---------------------------------------------------------------------------

def plot_dataset_composition(labels, ds_names, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    fig.suptitle("Test Set Composition", color="white", fontsize=13)

    for ax, subset_label, title in [
        (ax1, 1, "Real samples by dataset"),
        (ax2, 0, "Synthetic samples by dataset"),
    ]:
        ax.set_facecolor(DARK_BG)
        mask = labels == subset_label
        sub_ds = ds_names[mask]
        counts = pd.Series(sub_ds).value_counts()
        colors = DS_PALETTE[:len(counts)]
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            pctdistance=0.75,
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
        )
        for t in texts: t.set_color("white"); t.set_fontsize(9)
        for at in autotexts: at.set_color("white"); at.set_fontsize(8)
        ax.set_title(f"{title}\n(total: {mask.sum():,})", color="white", fontsize=10, pad=10)

    fig.tight_layout()
    fig.savefig(out_dir / "06_dataset_composition.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Score CDF (cumulative distribution)
# ---------------------------------------------------------------------------

def plot_score_cdf(labels, probs, out_dir):
    fig, ax = styled_fig(9, 5), None
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL_BG)

    for lbl, name, color in [(1, "Real", COLORS["real"]), (0, "Synthetic", COLORS["synth"])]:
        scores = np.sort(probs[labels == lbl])
        cdf = np.arange(1, len(scores)+1) / len(scores)
        ax.plot(scores, cdf, color=color, lw=2, label=f"{name}  (n={len(scores):,})")

    ax.axvline(0.5, color="#ffcc02", lw=1.5, linestyle="--", alpha=0.8, label="threshold = 0.5")
    ax.fill_betweenx([0,1], 0, 0.5, alpha=0.05, color=COLORS["synth"])
    ax.fill_betweenx([0,1], 0.5, 1, alpha=0.05, color=COLORS["real"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    style_ax(ax, "Cumulative Score Distribution  (test set)", "Realism Score", "Cumulative Fraction")
    ax.legend(facecolor=ACCENT, labelcolor="white", fontsize=10)
    for spine in ax.spines.values(): spine.set_edgecolor("#444466")
    ax.tick_params(colors="white")

    fig.tight_layout()
    fig.savefig(out_dir / "07_score_cdf.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Summary metrics table
# ---------------------------------------------------------------------------

def plot_summary_table(global_metrics, auc, ap, out_dir):
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG); ax.axis("off")
    ax.set_title("Global Test Set Metrics", color="white", fontsize=13, pad=14, fontweight="bold")

    metrics = [
        ("AUC-ROC",           f"{auc:.4f}"),
        ("Avg Precision (AP)", f"{ap:.4f}"),
        ("Accuracy",          f"{global_metrics['accuracy']:.4f}"),
        ("Precision",         f"{global_metrics['precision']:.4f}"),
        ("Recall",            f"{global_metrics['recall']:.4f}"),
        ("F1 Score",          f"{global_metrics['f1']:.4f}"),
        ("Threshold",         f"{global_metrics['threshold']:.2f}"),
    ]

    col_x = [0.15, 0.55]
    header_y = 0.88
    ax.text(col_x[0], header_y, "Metric", color="#cccccc", fontsize=11,
            fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(col_x[1], header_y, "Value", color="#cccccc", fontsize=11,
            fontweight="bold", transform=ax.transAxes, va="top")
    ax.plot([0.05, 0.95], [header_y - 0.08, header_y - 0.08],
            color="#444466", lw=0.8, transform=ax.transAxes)

    row_h = 0.11
    for i, (name, val) in enumerate(metrics):
        y = header_y - 0.14 - i * row_h
        bg = "#16213e" if i % 2 == 0 else "#0f3460"
        rect = plt.Rectangle((0.05, y - 0.05), 0.90, row_h,
                              transform=ax.transAxes, facecolor=bg, zorder=1)
        ax.add_patch(rect)
        ax.text(col_x[0], y, name, color="white", fontsize=10,
                transform=ax.transAxes, va="center", zorder=2)
        ax.text(col_x[1], y, val, color="#4fc3f7", fontsize=10,
                transform=ax.transAxes, va="center", fontweight="bold", zorder=2)

    fig.tight_layout()
    fig.savefig(out_dir / "08_summary_metrics.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Full dataset breakdown table (real + synth %)
# ---------------------------------------------------------------------------

def plot_dataset_breakdown_table(labels, probs, ds_names, threshold, out_dir):
    records = []
    total_n = len(labels)

    for ds in sorted(set(ds_names)):
        mask = ds_names == ds
        l, p = labels[mask], probs[mask]
        n = mask.sum()
        pct_of_total = n / total_n * 100
        true_cls = "Real" if l.mean() > 0.5 else "Synthetic"
        acc = float(((p >= threshold).astype(int) == l).mean())
        mean_s = float(p.mean())
        pct_above = float((p > threshold).mean() * 100)
        records.append({
            "dataset": ds,
            "true_class": true_cls,
            "n": n,
            "% of test": pct_of_total,
            "accuracy": acc,
            "mean_score": mean_s,
            "% > 0.5": pct_above,
        })

    df = pd.DataFrame(records)
    df = pd.concat([
        df[df["true_class"] == "Real"].sort_values("n", ascending=False),
        df[df["true_class"] == "Synthetic"].sort_values("n", ascending=False),
    ]).reset_index(drop=True)

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(13, 1.2 + n_rows * 0.52), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG); ax.axis("off")
    ax.set_title("Test Set — Dataset Breakdown", color="white", fontsize=13,
                 pad=14, fontweight="bold")

    cols     = ["Dataset", "Class", "Samples", "% of Test", "Accuracy", "Mean Score", "% > 0.5"]
    col_x    = [0.03, 0.30, 0.42, 0.52, 0.63, 0.75, 0.87]
    header_y = 0.93

    for x, col in zip(col_x, cols):
        ax.text(x, header_y, col, color="#cccccc", fontsize=9.5, fontweight="bold",
                transform=ax.transAxes, va="top")
    ax.plot([0.01, 0.99], [header_y - 0.05, header_y - 0.05],
            color="#444466", lw=0.8, transform=ax.transAxes)

    name_colors = {"Real": COLORS["real"], "Synthetic": COLORS["synth"]}
    row_colors  = ["#16213e", "#0f3460"]

    available_h = header_y - 0.1
    row_h = available_h / n_rows

    for i, row in df.iterrows():
        y_mid = header_y - 0.08 - (i + 0.5) * row_h
        y_bot = y_mid - row_h * 0.45
        rect = plt.Rectangle((0.01, y_bot), 0.98, row_h * 0.9,
                              transform=ax.transAxes,
                              facecolor=row_colors[i % 2], zorder=1)
        ax.add_patch(rect)
        vals = [
            row["dataset"],
            row["true_class"],
            f"{row['n']:,}",
            f"{row['% of test']:.1f}%",
            f"{row['accuracy']:.3f}",
            f"{row['mean_score']:.4f}",
            f"{row['% > 0.5']:.1f}%",
        ]
        for j, (x, v) in enumerate(zip(col_x, vals)):
            color = name_colors.get(row["true_class"], "white") if j == 0 else \
                    (name_colors.get(row["true_class"], "white") if j == 1 else "white")
            ax.text(x, y_mid, v, color=color, fontsize=9,
                    transform=ax.transAxes, va="center", zorder=2)

    fig.tight_layout()
    fig.savefig(out_dir / "09_dataset_breakdown.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test set analysis and visualization")
    parser.add_argument("--config",     required=True,  help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", default="outputs/test_analysis")
    args = parser.parse_args()

    from realism_classifier.config import load_config
    from realism_classifier.dataset import build_dataloaders
    from realism_classifier.evaluate import collect_predictions, compute_metrics
    from realism_classifier.model import load_model_from_checkpoint
    from realism_classifier.config import resolve_device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading config and model...")
    config = load_config(args.config)
    device = resolve_device(config.device)
    model  = load_model_from_checkpoint(args.checkpoint, device=device)

    print("Building test dataloader...")
    loaders = build_dataloaders(config.data, config.training, splits=("test",))
    if "test" not in loaders:
        print("ERROR: no 'test' split found in data.")
        sys.exit(1)

    loader = loaders["test"]
    print(f"  Test samples: {len(loader.dataset):,}")

    print("Running inference on test set...")
    probs, preds, labels = collect_predictions(
        model, loader, device, threshold=config.evaluation.threshold
    )

    meta     = loader.dataset.get_metadata()
    raw_ds   = meta["dataset"].values if "dataset" in meta.columns else np.full(len(probs), "unknown")
    ds_names = remap_datasets(raw_ds)

    print("\nComputing global metrics...")
    global_m = compute_metrics(labels, probs, threshold=config.evaluation.threshold)
    for k, v in global_m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nGenerating plots...")

    print("  [1/9] ROC + PR curves")
    auc, ap = plot_roc_pr(labels, probs, out_dir)

    print("  [2/9] Score distribution (global)")
    plot_score_dist_global(labels, probs, out_dir)

    print("  [3/9] Per-dataset violin plot")
    plot_per_dataset_violin(labels, probs, ds_names, out_dir)

    print("  [4/9] Confusion matrix")
    plot_confusion_matrix(labels, probs, config.evaluation.threshold, out_dir)

    print("  [5/9] Per-dataset accuracy bar")
    acc_df = plot_per_dataset_accuracy(labels, probs, ds_names, config.evaluation.threshold, out_dir)

    print("  [6/9] Dataset composition pie charts")
    plot_dataset_composition(labels, ds_names, out_dir)

    print("  [7/9] Score CDF")
    plot_score_cdf(labels, probs, out_dir)

    print("  [8/9] Summary metrics table")
    plot_summary_table(global_m, auc, ap, out_dir)

    print("  [9/9] Dataset breakdown table")
    breakdown_df = plot_dataset_breakdown_table(labels, probs, ds_names, config.evaluation.threshold, out_dir)

    # Save CSVs
    breakdown_df.to_csv(out_dir / "dataset_breakdown.csv", index=False)
    pd.DataFrame([{**global_m, "auc_roc": auc, "avg_precision": ap}]).to_csv(
        out_dir / "global_metrics.csv", index=False
    )

    print(f"\nAll outputs saved to: {out_dir}/")
    print("\nFiles generated:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
