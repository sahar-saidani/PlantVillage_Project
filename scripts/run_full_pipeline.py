from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.classical_ml import train_and_evaluate_classical
from plant_disease.config import load_config
from plant_disease.dataset import build_index, save_metadata, stratified_split
from plant_disease.deep_learning import train_and_evaluate_deep
from plant_disease.utils import ensure_dir, set_seed, write_json
from plant_disease.visualization import generate_examples


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

    examples = generate_examples(config, metadata_path)
    classical = train_and_evaluate_classical(config, metadata_path)
    deep = train_and_evaluate_deep(config, metadata_path)

    summary = {
        "metadata_path": str(metadata_path),
        "num_examples_generated": len(examples),
        "classical_models": list(classical.keys()),
        "deep_model": deep["model_name"],
    }
    write_json(config.paths.metadata_dir / "run_summary.json", summary)
    print("Full pipeline completed.")


if __name__ == "__main__":
    main()
