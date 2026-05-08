"""CLI entry point for inference (scoring new embeddings)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from realism_classifier.config import load_config, merge_cli_overrides, parse_overrides
from realism_classifier.inference import score_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a new embedding dataset with a trained realism classifier."
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--embeddings", default=None, metavar="PATH",
        help="Path to embeddings file (.npy or .npz)"
    )
    parser.add_argument(
        "--metadata", default=None, metavar="PATH",
        help="Path to metadata CSV (optional; no labels required)"
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Output CSV path for realism scores"
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

    out_df = score_dataset(
        config,
        embeddings_path=args.embeddings,
        metadata_csv=args.metadata,
        output_csv=args.output,
        checkpoint_path=args.checkpoint,
    )

    print(f"\nScored {len(out_df)} samples.")
    print(f"Mean realism score: {out_df['realism_score'].mean():.4f}")
    print(out_df[["sample_id", "realism_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
