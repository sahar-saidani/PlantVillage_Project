from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.classical_ml import train_and_evaluate_classical
from plant_disease.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_path = config.paths.metadata_dir / "dataset_index.csv"
    results = train_and_evaluate_classical(config, metadata_path)
    print(f"Classical ML results saved to {config.paths.classical_dir}")
    print(results)


if __name__ == "__main__":
    main()

