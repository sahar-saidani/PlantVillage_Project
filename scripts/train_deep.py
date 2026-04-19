from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.config import load_config
from plant_disease.deep_learning import train_and_evaluate_deep
from plant_disease.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    metadata_path = config.paths.metadata_dir / "dataset_index.csv"
    results = train_and_evaluate_deep(config, metadata_path)
    print(f"Deep learning results saved to {config.paths.deep_dir}")
    print(results)


if __name__ == "__main__":
    main()

