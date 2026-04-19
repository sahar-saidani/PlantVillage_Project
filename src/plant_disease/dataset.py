from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

from .config import AppConfig
from .utils import write_csv

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def build_index(config: AppConfig) -> list[dict[str, str]]:
    dataset_root = config.paths.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. "
            "Extract the Kaggle dataset under data/raw/PlantVillage."
        )

    rows: list[dict[str, str]] = []
    for label in config.dataset.classes:
        class_dir = dataset_root / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        for image_path in list_images(class_dir):
            rows.append({"image_path": str(image_path), "label": label})
    if not rows:
        raise RuntimeError("No images found in configured dataset folders.")
    return rows


def stratified_split(
    rows: list[dict[str, str]], train_ratio: float, val_ratio: float, seed: int
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    split_rows: list[dict[str, str]] = []
    for label, label_rows in grouped.items():
        rng.shuffle(label_rows)
        n_total = len(label_rows)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        for idx, row in enumerate(label_rows):
            split = "test"
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            split_rows.append(
                {
                    "image_path": row["image_path"],
                    "label": label,
                    "split": split,
                }
            )
    return split_rows


def save_metadata(config: AppConfig, rows: list[dict[str, str]]) -> Path:
    output_path = config.paths.metadata_dir / "dataset_index.csv"
    write_csv(output_path, rows, ["image_path", "label", "split"])
    return output_path

