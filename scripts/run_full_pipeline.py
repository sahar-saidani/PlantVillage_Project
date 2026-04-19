from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.classical_ml import train_and_evaluate_classical
from plant_disease.config import load_config
from plant_disease.dataset import build_index, save_metadata, stratified_split
from plant_disease.deep_learning import reuse_or_evaluate_saved_deep_model, train_and_evaluate_deep
from plant_disease.segmentation_benchmark import benchmark_segmentation_methods, save_segmentation_benchmark
from plant_disease.utils import ensure_dir, set_seed, write_json
from plant_disease.visualization import generate_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--force-train-deep",
        action="store_true",
        help="Force local deep training even if artifacts/deep_learning/best_model.pt already exists.",
    )
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
    segmentation = benchmark_segmentation_methods(config, split_rows)
    save_segmentation_benchmark(config, segmentation)
    classical = train_and_evaluate_classical(config, metadata_path)
    deep_model_path = config.paths.deep_dir / "best_model.pt"
    reused_existing_deep_model = deep_model_path.exists() and not args.force_train_deep
    if reused_existing_deep_model:
        deep = reuse_or_evaluate_saved_deep_model(config, metadata_path)
    else:
        deep = train_and_evaluate_deep(config, metadata_path)

    summary = {
        "metadata_path": str(metadata_path),
        "num_examples_generated": len(examples),
        "best_segmentation_method": segmentation["best_method"],
        "classical_models": list(classical.keys()),
        "deep_model": deep["model_name"],
        "reused_existing_deep_model": reused_existing_deep_model,
    }
    write_json(config.paths.metadata_dir / "run_summary.json", summary)
    print("Full pipeline completed.")


if __name__ == "__main__":
    main()
