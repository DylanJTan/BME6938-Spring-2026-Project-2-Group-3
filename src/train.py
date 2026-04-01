"""Training entrypoint scaffold for BME6938 Project 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train model scaffold")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--model", type=str, required=True, help="Model key to train")
    return parser.parse_args()


def main() -> None:
    """Load config and print scaffold status for next implementation step."""
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("[Scaffold] Train entrypoint reached.")
    print(f"Model requested: {args.model}")
    print(f"Dataset root: {config['paths']['dataset_root']}")
    print("TODO: Implement data indexing, splits, model factory, training loop, and checkpointing.")


if __name__ == "__main__":
    main()
