from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .preprocessing import advanced_leaf_mask, kmeans_mask, leaf_mask, read_image, threshold_mask
from .utils import write_json


def _mask_quality(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask_bin = mask > 0
    foreground = int(np.count_nonzero(mask_bin))
    total = int(mask_bin.size)
    if foreground == 0:
        return {
            "area_ratio": 0.0,
            "green_contrast": 0.0,
            "largest_component_ratio": 0.0,
            "edge_alignment": 0.0,
            "quality_score": 0.0,
        }

    rgb = image.astype(np.float32)
    excess_green = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    fg_green = float(excess_green[mask_bin].mean())
    bg_green = float(excess_green[~mask_bin].mean()) if np.any(~mask_bin) else 0.0
    green_contrast = max(0.0, (fg_green - bg_green) / 255.0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        largest_component_ratio = 0.0
    else:
        largest_component = int(stats[1:, cv2.CC_STAT_AREA].max())
        largest_component_ratio = float(largest_component / max(foreground, 1))

    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 80, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    boundary_pixels = int(np.count_nonzero(boundary))
    overlap = int(np.count_nonzero((boundary > 0) & (edges > 0)))
    edge_alignment = float(overlap / boundary_pixels) if boundary_pixels else 0.0

    area_ratio = float(foreground / total)
    area_plausibility = max(0.0, 1.0 - abs(area_ratio - 0.45) / 0.45)
    quality_score = float(
        0.35 * green_contrast
        + 0.25 * largest_component_ratio
        + 0.20 * edge_alignment
        + 0.20 * area_plausibility
    )

    return {
        "area_ratio": area_ratio,
        "green_contrast": green_contrast,
        "largest_component_ratio": largest_component_ratio,
        "edge_alignment": edge_alignment,
        "quality_score": quality_score,
    }


def benchmark_segmentation_methods(
    config: AppConfig,
    rows: list[dict[str, str]],
    max_samples_per_class: int = 12,
) -> dict[str, object]:
    selected_rows: list[dict[str, str]] = []
    per_class_counts = {label: 0 for label in config.dataset.classes}
    for row in rows:
        if row["split"] != "train":
            continue
        label = row["label"]
        if label not in per_class_counts:
            continue
        if per_class_counts[label] >= max_samples_per_class:
            continue
        selected_rows.append(row)
        per_class_counts[label] += 1

    methods = {
        "threshold_otsu": lambda image: threshold_mask(image),
        "hsv_mask": lambda image: leaf_mask(image),
        "kmeans_mask": lambda image: kmeans_mask(image),
        "advanced_hybrid_grabcut": lambda image: advanced_leaf_mask(image),
    }
    descriptions = {
        "threshold_otsu": "Global thresholding on grayscale after classical preprocessing.",
        "hsv_mask": "Classical HSV green segmentation with morphology.",
        "kmeans_mask": "Color clustering in RGB space with morphology.",
        "advanced_hybrid_grabcut": (
            "Advanced pipeline: bilateral + CLAHE + sharpening, HSV/KMeans seeds, then GrabCut refinement."
        ),
    }

    aggregate = {
        name: {
            "quality_score": [],
            "green_contrast": [],
            "largest_component_ratio": [],
            "edge_alignment": [],
            "area_ratio": [],
        }
        for name in methods
    }

    for row in selected_rows:
        image = read_image(row["image_path"], config.dataset.image_size)
        for name, method in methods.items():
            mask = method(image)
            stats = _mask_quality(image, mask)
            for key, value in stats.items():
                aggregate[name][key].append(value)

    results = {}
    for name, metrics in aggregate.items():
        results[name] = {
            "description": descriptions[name],
            **{
                metric_name: float(np.mean(values)) if values else 0.0
                for metric_name, values in metrics.items()
            },
        }

    ranking = sorted(
        (
            {"method": name, "quality_score": stats["quality_score"]}
            for name, stats in results.items()
        ),
        key=lambda item: item["quality_score"],
        reverse=True,
    )

    return {
        "num_images_evaluated": len(selected_rows),
        "per_class_limit": max_samples_per_class,
        "methods": results,
        "ranking": ranking,
        "best_method": ranking[0]["method"] if ranking else None,
        "notes": [
            "This benchmark is unsupervised: it scores segmentation quality using green contrast, component coherence, boundary alignment, and plausible leaf coverage.",
            "It is meant to compare preprocessing/segmentation strategies when no pixel-level masks are available.",
        ],
    }


def save_segmentation_benchmark(config: AppConfig, payload: dict[str, object]) -> Path:
    output_path = config.paths.metadata_dir / "segmentation_benchmark.json"
    write_json(output_path, payload)
    return output_path
