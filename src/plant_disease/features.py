from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import apply_mask, edge_map, leaf_mask, preprocess_image


def _safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _glcm_features(gray: np.ndarray, levels: int = 16) -> list[float]:
    scaled = (gray.astype(np.float32) / 256.0 * levels).astype(np.int32)
    scaled = np.clip(scaled, 0, levels - 1)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    left = scaled[:, :-1]
    right = scaled[:, 1:]

    for i in range(levels):
        for j in range(levels):
            glcm[i, j] = np.sum((left == i) & (right == j))

    total = glcm.sum()
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0]

    glcm /= total
    ii, jj = np.indices(glcm.shape)
    contrast = float(np.sum(glcm * (ii - jj) ** 2))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(ii - jj))))
    energy = float(np.sum(glcm**2))
    mean_i = np.sum(ii * glcm)
    mean_j = np.sum(jj * glcm)
    std_i = np.sqrt(np.sum(glcm * (ii - mean_i) ** 2))
    std_j = np.sqrt(np.sum(glcm * (jj - mean_j) ** 2))
    denom = float(std_i * std_j) if std_i > 0 and std_j > 0 else 1.0
    correlation = float(np.sum(glcm * (ii - mean_i) * (jj - mean_j)) / denom)
    return [contrast, homogeneity, energy, correlation]


def extract_features(image: np.ndarray, hsv_bins: int) -> np.ndarray:
    preprocessed = preprocess_image(image)
    mask = leaf_mask(preprocessed)
    masked = apply_mask(preprocessed, mask)
    edges = edge_map(preprocessed)

    rgb_pixels = masked[mask > 0]
    hsv = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv[mask > 0]
    gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

    features: list[float] = []

    for channel in range(3):
        mean_val, std_val = _safe_mean_std(rgb_pixels[:, channel] if rgb_pixels.size else np.array([]))
        features.extend([mean_val, std_val])

    for channel in range(3):
        hist = cv2.calcHist([masked], [channel], mask, [hsv_bins], [0, 256]).flatten()
        hist = hist / max(hist.sum(), 1.0)
        features.extend(hist.tolist())

    for channel in range(3):
        mean_val, std_val = _safe_mean_std(hsv_pixels[:, channel] if hsv_pixels.size else np.array([]))
        features.extend([mean_val, std_val])

    hist = cv2.calcHist([hsv], [0, 1], mask, [hsv_bins, hsv_bins], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    features.extend(hist.tolist())

    lesion_ratio = float(1.0 - (np.count_nonzero(mask) / mask.size))
    edge_density = float(np.count_nonzero(edges) / edges.size)
    features.extend([lesion_ratio, edge_density])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    if largest is None:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        area = float(cv2.contourArea(largest))
        perimeter = float(cv2.arcLength(largest, True))
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = float(w / max(h, 1))
        circularity = float((4.0 * np.pi * area) / (perimeter**2)) if perimeter > 0 else 0.0
        hu = cv2.HuMoments(cv2.moments(largest)).flatten()
        features.extend([area, perimeter, aspect_ratio, circularity, float(np.sum(np.abs(hu)))])

    features.extend(_glcm_features(gray))
    return np.asarray(features, dtype=np.float32)
