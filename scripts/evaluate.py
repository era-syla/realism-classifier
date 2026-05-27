"""
Evaluate a trained realism classifier on the labelled test set.

Generates 9 diagnostic plots and summary CSVs:
  01_roc_pr.png               ROC and Precision-Recall curves
  02_score_dist_global.png    Score distribution (real vs synthetic)
  03_per_dataset_violin.png   Per-dataset score violin plot
  04_confusion_matrix.png     Confusion matrix (row-normalised %)
  05_per_dataset_accuracy.png Per-dataset accuracy bar chart
  06_dataset_composition.png  Test set composition pie charts
  07_score_cdf.png            Cumulative score distribution
  08_summary_metrics.png      Global metrics table
  09_dataset_breakdown.png    Per-dataset breakdown table
  dataset_breakdown.csv
  global_metrics.csv

Usage:
    python scripts/evaluate.py \
        --config     configs/default.yaml \
        --checkpoint outputs/runs/run_03/best_model.pt \
        --output-dir outputs/test_analysis/run_03
"""

import argparse
import sys
from pathlib import Path

# reuse the full test_analysis implementation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# import everything from test_analysis
import importlib.util, os

_ta_path = Path(__file__).resolve().parent / "test_analysis.py"
_spec    = importlib.util.spec_from_file_location("test_analysis", _ta_path)
_ta      = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ta)


def main():
    parser = argparse.ArgumentParser(description="Full test-set evaluation with plots.")
    parser.add_argument("--config",     required=True,  help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", default="outputs/test_analysis")
    args = parser.parse_args()

    # delegate to test_analysis.main() with the same args format
    sys.argv = [
        "evaluate.py",
        "--config",     args.config,
        "--checkpoint", args.checkpoint,
        "--output-dir", args.output_dir,
    ]
    _ta.main()


if __name__ == "__main__":
    main()
