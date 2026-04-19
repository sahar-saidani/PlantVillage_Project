from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.config import load_config
from plant_disease.dataset import build_index, save_metadata, stratified_split
from plant_disease.utils import ensure_dir, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    ensure_dir(config.paths.metadata_dir)

    rows = build_index(config)
    split_rows = stratified_split(
        rows,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        seed=config.seed,
    )
    metadata_path = save_metadata(config, split_rows)

    summary = {
        "num_images": len(split_rows),
        "classes": config.dataset.classes,
        "metadata_path": str(metadata_path),
    }
    write_json(config.paths.metadata_dir / "dataset_summary.json", summary)
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()

