"""CLI entry point for training."""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from realism_classifier.config import load_config, merge_cli_overrides, parse_overrides
from realism_classifier.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train the realism classifier.")
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to YAML config file (e.g. configs/default.yaml)"
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Dot-notation override, e.g. --override training.learning_rate=3e-4 (repeatable)"
    )
    parser.add_argument(
        "--run-name", default=None, metavar="NAME",
        help="Subdirectory name under checkpointing.output_dir; defaults to timestamp"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.override:
        overrides = parse_overrides(args.override)
        config = merge_cli_overrides(config, overrides)

    run_dir = None
    if args.run_name:
        from pathlib import Path
        run_dir = str(Path(config.checkpointing.output_dir) / args.run_name)

    best_ckpt = train(config, run_dir=run_dir)
    print(f"\nBest checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
