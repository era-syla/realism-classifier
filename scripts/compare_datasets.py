"""CLI entry point for cross-dataset realism score comparison."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from realism_classifier.config import load_config, merge_cli_overrides, parse_overrides
from realism_classifier.compare import compare_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare realism score distributions across multiple datasets."
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--scores-dir", default=None, metavar="PATH",
        help="Directory containing per-dataset inference CSVs; "
             "overrides config.comparison.scores_dir"
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="PATH",
        help="Output directory for ranking CSV and plots; "
             "overrides config.comparison.output_dir"
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Only include top-k datasets in violin/bar plots"
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Dot-notation config override (repeatable)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.override:
        config = merge_cli_overrides(config, parse_overrides(args.override))
    if args.scores_dir:
        config.comparison.scores_dir = args.scores_dir
    if args.output_dir:
        config.comparison.output_dir = args.output_dir
    if args.top_k is not None:
        config.comparison.top_k = args.top_k

    result = compare_datasets(config)

    print(f"\nRanking saved to: {result['ranking_csv']}")
    print(f"Plots saved to:   {result['output_dir']}")


if __name__ == "__main__":
    main()
