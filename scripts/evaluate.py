"""CLI entry point for evaluation on a labeled split."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from realism_classifier.config import load_config, merge_cli_overrides, parse_overrides
from realism_classifier.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained realism classifier.")
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Path to model checkpoint (.pt); overrides config.inference.checkpoint_path"
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which split to evaluate (default: test)"
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="PATH",
        help="Directory to write evaluation results and plots"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Probability threshold for binary decision (default: from config)"
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
    if args.threshold is not None:
        config.evaluation.threshold = args.threshold

    results = evaluate(
        config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
    )

    print("\n=== Global Metrics ===")
    for k, v in results["global_metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if not results["per_dataset_metrics"].empty:
        print("\n=== Per-Dataset Metrics ===")
        print(results["per_dataset_metrics"].to_string())


if __name__ == "__main__":
    main()
