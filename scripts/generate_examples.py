from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.config import load_config
from plant_disease.visualization import generate_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_path = config.paths.metadata_dir / "dataset_index.csv"
    output_paths = generate_examples(config, metadata_path)
    print(f"Generated {len(output_paths)} example visualizations in {config.paths.examples_dir}")


if __name__ == "__main__":
    main()

