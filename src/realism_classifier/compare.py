"""Cross-dataset realism score comparison, summary statistics, and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scored_csvs(scores_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from scores_dir and concatenate into one DataFrame.

    Each file must have at minimum a 'realism_score' column and ideally a
    'sample_id' column. If no 'dataset' column is present, the filename stem
    is used as the dataset name.

    Returns a single concatenated DataFrame.
    """
    p = Path(scores_dir)
    if not p.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_dir}")

    csv_files = sorted(p.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {scores_dir}")

    frames: List[pd.DataFrame] = []
    for f in csv_files:
        df = pd.read_csv(str(f))
        if "realism_score" not in df.columns:
            print(f"Warning: {f.name} has no 'realism_score' column — skipping.")
            continue
        if "dataset" not in df.columns:
            df["dataset"] = f.stem
        frames.append(df)

    if not frames:
        raise ValueError("No valid scored CSVs found (all were missing 'realism_score').")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_dataset_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by 'dataset' and compute per-dataset statistics.

    Columns: mean, median, std, min, max, p10, p25, p75, p90, count
    Sorted descending by mean realism score.
    """
    records = []
    for ds, grp in scored_df.groupby("dataset"):
        scores = grp["realism_score"].values
        records.append(
            {
                "dataset": ds,
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "p10": float(np.percentile(scores, 10)),
                "p25": float(np.percentile(scores, 25)),
                "p75": float(np.percentile(scores, 75)),
                "p90": float(np.percentile(scores, 90)),
                "count": int(len(scores)),
            }
        )
    df = pd.DataFrame(records).set_index("dataset")
    return df.sort_values("mean", ascending=False)


def rank_datasets(
    summary_df: pd.DataFrame,
    by: str = "mean",
) -> pd.DataFrame:
    """
    Return summary_df sorted descending by `by` with a 1-based 'rank' column.
    """
    ranked = summary_df.sort_values(by, ascending=False).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_histograms(
    scored_df: pd.DataFrame,
    output_dir: str,
    plot_format: str = "png",
    bins: int = 50,
) -> None:
    """
    One histogram per dataset saved to output_dir/hist_{dataset}.{format}.
    """
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ds, grp in scored_df.groupby("dataset"):
        scores = grp["realism_score"].values
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scores, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Realism Score")
        ax.set_ylabel("Count")
        ax.set_title(f"{ds}  (n={len(scores):,})")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fname = f"hist_{str(ds).replace('/', '_')}.{plot_format}"
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def plot_comparison_boxplot(
    scored_df: pd.DataFrame,
    output_path: str,
    top_k: Optional[int] = None,
    plot_format: str = "png",
) -> None:
    """
    Side-by-side violin + box plot of realism score distributions.
    Datasets are ordered left-to-right by descending median score.
    If top_k is given, only the top-k by median are shown.
    """
    import matplotlib.pyplot as plt

    medians = scored_df.groupby("dataset")["realism_score"].median().sort_values(ascending=False)
    if top_k is not None:
        medians = medians.head(top_k)

    datasets = medians.index.tolist()
    data = [scored_df.loc[scored_df["dataset"] == ds, "realism_score"].values for ds in datasets]

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 0.8), 5))
    parts = ax.violinplot(data, positions=range(len(datasets)), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Realism Score")
    ax.set_ylim(0, 1)
    title = "Dataset Realism Score Distributions"
    if top_k is not None:
        title += f" (top {top_k})"
    ax.set_title(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ranking_bar(
    summary_df: pd.DataFrame,
    output_path: str,
    plot_format: str = "png",
) -> None:
    """
    Horizontal bar chart of mean realism score per dataset, sorted descending.
    Error bars show ± std.
    """
    import matplotlib.pyplot as plt

    df = summary_df.sort_values("mean", ascending=True)  # ascending for horizontal bar
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    ax.barh(
        df.index,
        df["mean"],
        xerr=df["std"],
        color="steelblue",
        ecolor="gray",
        capsize=3,
        alpha=0.85,
    )
    ax.set_xlabel("Mean Realism Score")
    ax.set_xlim(0, 1)
    ax.set_title("Dataset Ranking by Mean Realism Score")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def compare_datasets(config) -> Dict:
    """
    Full comparison pipeline.

    1. Load all scored CSVs from config.comparison.scores_dir
    2. Compute per-dataset summary statistics
    3. Rank datasets
    4. Generate plots (histograms, violin/box, ranking bar)
    5. Save dataset_ranking.csv to output_dir
    6. Return dict with summary DataFrame and output paths
    """
    out = Path(config.comparison.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scored_df = load_scored_csvs(config.comparison.scores_dir)
    summary = compute_dataset_summary(scored_df)
    ranked = rank_datasets(summary)

    ranking_csv = out / "dataset_ranking.csv"
    ranked.reset_index().to_csv(ranking_csv, index=False)
    print(f"Dataset ranking saved to {ranking_csv}")
    print(ranked[["rank", "mean", "median", "std", "count"]].to_string())

    fmt = config.comparison.plot_format
    top_k = config.comparison.top_k

    try:
        plot_score_histograms(scored_df, str(out / "histograms"), plot_format=fmt)
        boxplot_path = str(out / f"comparison_violin.{fmt}")
        plot_comparison_boxplot(scored_df, boxplot_path, top_k=top_k, plot_format=fmt)
        ranking_bar_path = str(out / f"ranking_bar.{fmt}")
        plot_ranking_bar(summary, ranking_bar_path, plot_format=fmt)
    except Exception as exc:
        print(f"Warning: could not generate one or more plots: {exc}")

    return {
        "summary": ranked,
        "scored_df": scored_df,
        "ranking_csv": str(ranking_csv),
        "output_dir": str(out),
    }
