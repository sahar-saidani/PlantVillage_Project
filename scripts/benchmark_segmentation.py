from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.config import load_config
from plant_disease.segmentation_benchmark import benchmark_segmentation_methods, save_segmentation_benchmark
from plant_disease.utils import read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--per-class", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_path = config.paths.metadata_dir / "dataset_index.csv"
    rows = read_csv(metadata_path)
    payload = benchmark_segmentation_methods(config, rows, max_samples_per_class=args.per_class)
    output_path = save_segmentation_benchmark(config, payload)
    print(f"Segmentation benchmark saved to {output_path}")


if __name__ == "__main__":
    main()
