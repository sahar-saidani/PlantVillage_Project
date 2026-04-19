from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .preprocessing import (
    advanced_leaf_mask,
    advanced_preprocess_image,
    apply_mask,
    contour_overlay,
    edge_map,
    leaf_mask,
    preprocess_image,
    read_image,
)
from .utils import ensure_dir, read_csv


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def generate_examples(config: AppConfig, metadata_path: Path, per_class: int = 2) -> list[Path]:
    rows = [row for row in read_csv(metadata_path) if row["split"] == "train"]
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["label"], []).append(row)

    output_paths: list[Path] = []
    ensure_dir(config.paths.examples_dir)
    for label in config.dataset.classes:
        samples = by_class.get(label, [])[:per_class]
        for idx, row in enumerate(samples, start=1):
            original = read_image(row["image_path"], config.dataset.image_size)
            preprocessed = preprocess_image(original)
            advanced_preprocessed = advanced_preprocess_image(original)
            mask = leaf_mask(preprocessed)
            advanced_mask = advanced_leaf_mask(original)
            masked = apply_mask(preprocessed, mask)
            advanced_masked = apply_mask(advanced_preprocessed, advanced_mask)
            edges = edge_map(preprocessed)
            contours = contour_overlay(preprocessed, edges)

            grid = np.concatenate(
                [
                    _to_bgr(original),
                    _to_bgr(preprocessed),
                    _to_bgr(advanced_preprocessed),
                    _to_bgr(mask),
                    _to_bgr(advanced_mask),
                    _to_bgr(masked),
                    _to_bgr(advanced_masked),
                    _to_bgr(edges),
                    _to_bgr(contours),
                ],
                axis=1,
            )
            output_path = config.paths.examples_dir / f"{label}_sample_{idx}.png"
            cv2.imwrite(str(output_path), grid)
            output_paths.append(output_path)
    return output_paths
