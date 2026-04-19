from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectPaths:
    dataset_root: Path
    metadata_dir: Path
    examples_dir: Path
    classical_dir: Path
    deep_dir: Path


@dataclass
class DatasetConfig:
    image_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    classes: list[str]


@dataclass
class ClassicalConfig:
    hsv_bins: int
    random_forest_trees: int
    svm_c: float


@dataclass
class DeepConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    model_name: str
    use_pretrained_if_available: bool


@dataclass
class AppConfig:
    seed: int
    paths: ProjectPaths
    dataset: DatasetConfig
    classical_ml: ClassicalConfig
    deep_learning: DeepConfig


def _to_paths(base_dir: Path, raw_paths: dict[str, Any]) -> ProjectPaths:
    return ProjectPaths(
        dataset_root=(base_dir / raw_paths["dataset_root"]).resolve(),
        metadata_dir=(base_dir / raw_paths["metadata_dir"]).resolve(),
        examples_dir=(base_dir / raw_paths["examples_dir"]).resolve(),
        classical_dir=(base_dir / raw_paths["classical_dir"]).resolve(),
        deep_dir=(base_dir / raw_paths["deep_dir"]).resolve(),
    )


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).resolve()
    base_dir = config_path.parent.parent
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    ratios = (
        data["dataset"]["train_ratio"]
        + data["dataset"]["val_ratio"]
        + data["dataset"]["test_ratio"]
    )
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    return AppConfig(
        seed=int(data["seed"]),
        paths=_to_paths(base_dir, data["paths"]),
        dataset=DatasetConfig(**data["dataset"]),
        classical_ml=ClassicalConfig(**data["classical_ml"]),
        deep_learning=DeepConfig(**data["deep_learning"]),
    )

