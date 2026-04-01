"""Evaluation entrypoint scaffold for BME6938 Project 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model scaffold")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    return parser.parse_args()


def main() -> None:
    """Load config and print scaffold status for next implementation step."""
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("[Scaffold] Evaluate entrypoint reached.")
    print(f"Checkpoint requested: {args.checkpoint}")
    print(f"Output directory: {config['paths']['output_dir']}")
    print("TODO: Implement checkpoint loading, test dataloader, and metrics/plot export.")


if __name__ == "__main__":
    main()
