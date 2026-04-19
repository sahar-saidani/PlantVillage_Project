from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .preprocessing import preprocess_image


def read_rgb_image(path: str | Path, image_size: int = 224) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def resize_rgb_image(image: np.ndarray, image_size: int = 224) -> np.ndarray:
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def threshold_mask(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def hsv_mask(image: np.ndarray) -> np.ndarray:
    hsv = to_hsv(image)
    lower_green = np.array([20, 20, 20], dtype=np.uint8)
    upper_green = np.array([110, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower_green, upper_green)


def kmeans_mask(image: np.ndarray, k: int = 3) -> np.ndarray:
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    green_scores = centers[:, 1].astype(np.int32) - (
        centers[:, 0].astype(np.int32) + centers[:, 2].astype(np.int32)
    ) / 2
    leaf_cluster = int(np.argmax(green_scores))
    return (labels.flatten() == leaf_cluster).astype(np.uint8).reshape(image.shape[:2]) * 255


def sobel_edges(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canny_edges(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return cv2.Canny(gray, 80, 160)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, image, mask=mask)


def glcm_features(gray: np.ndarray, levels: int = 16) -> list[float]:
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
    homogeneity = float(np.sum(glcm / (1 + np.abs(ii - jj))))
    energy = float(np.sum(glcm**2))
    mean_i = np.sum(ii * glcm)
    mean_j = np.sum(jj * glcm)
    std_i = np.sqrt(np.sum(glcm * (ii - mean_i) ** 2))
    std_j = np.sqrt(np.sum(glcm * (jj - mean_j) ** 2))
    denom = float(std_i * std_j) if std_i > 0 and std_j > 0 else 1.0
    correlation = float(np.sum(glcm * (ii - mean_i) * (jj - mean_j)) / denom)
    return [contrast, homogeneity, energy, correlation]


def shape_features(mask: np.ndarray) -> list[float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    if largest is None:
        return [0.0, 0.0, 0.0]
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, True))
    circularity = float((4.0 * np.pi * area) / (perimeter**2)) if perimeter > 0 else 0.0
    return [area, perimeter, circularity]


def extract_classical_features(image: np.ndarray, bins: int = 16) -> np.ndarray:
    preprocessed = preprocess_image(image)
    hsv = to_hsv(preprocessed)
    gray = to_gray(preprocessed)

    mask_hsv = hsv_mask(preprocessed)
    mask_otsu = threshold_mask(preprocessed)
    mask_kmeans = kmeans_mask(preprocessed)
    sobel = sobel_edges(preprocessed)
    canny = canny_edges(preprocessed)

    rgb_hist: list[float] = []
    for channel in range(3):
        hist = cv2.calcHist([preprocessed], [channel], None, [bins], [0, 256]).flatten()
        rgb_hist.extend((hist / max(hist.sum(), 1.0)).tolist())

    hsv_hist: list[float] = []
    ranges = [(0, 180), (0, 256), (0, 256)]
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [bins], list(ranges[channel])).flatten()
        hsv_hist.extend((hist / max(hist.sum(), 1.0)).tolist())

    texture = glcm_features(gray)
    shape = shape_features(mask_hsv)
    extra = [
        float(np.count_nonzero(mask_otsu) / mask_otsu.size),
        float(np.count_nonzero(mask_hsv) / mask_hsv.size),
        float(np.count_nonzero(mask_kmeans) / mask_kmeans.size),
        float(np.count_nonzero(sobel) / sobel.size),
        float(np.count_nonzero(canny) / canny.size),
    ]
    return np.asarray(rgb_hist + hsv_hist + texture + shape + extra, dtype=np.float32)


def build_visual_pipeline(image: np.ndarray) -> dict[str, np.ndarray]:
    preprocessed = preprocess_image(image)
    otsu = threshold_mask(preprocessed)
    hsv = hsv_mask(preprocessed)
    kmeans = kmeans_mask(preprocessed)
    sobel = sobel_edges(preprocessed)
    canny = canny_edges(preprocessed)
    return {
        "original": image,
        "preprocessed": preprocessed,
        "grayscale": to_gray(preprocessed),
        "threshold_mask": otsu,
        "hsv_mask": hsv,
        "kmeans_mask": kmeans,
        "sobel_edges": sobel,
        "canny_edges": canny,
        "segmented_leaf": apply_mask(preprocessed, hsv),
    }


@dataclass
class ClassicalInference:
    model: Any
    classes: list[str]
    bins: int = 16

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        features = extract_classical_features(image, bins=self.bins).reshape(1, -1)
        pred_idx = int(self.model.predict(features)[0])
        scores = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            scores = {self.classes[i]: float(proba[i]) for i in range(len(self.classes))}
        return {
            "label": self.classes[pred_idx],
            "scores": scores,
            "feature_dim": int(features.shape[1]),
        }


def load_classical_inference(
    model_path: str | Path, class_names_path: str | Path | None = None
) -> ClassicalInference:
    model_path = Path(model_path)
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    if class_names_path and Path(class_names_path).exists():
        classes = json.loads(Path(class_names_path).read_text(encoding="utf-8"))
    else:
        classes = [
            "Tomato_healthy",
            "Tomato_Early_blight",
            "Tomato_Late_blight",
            "Tomato_Bacterial_spot",
        ]
    return ClassicalInference(model=model, classes=classes)


class DeepInference:
    def __init__(self, model: Any, classes: list[str], image_size: int = 224) -> None:
        self.model = model
        self.classes = classes
        self.image_size = image_size

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        import torch

        tensor = self._to_tensor(image)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return {
            "label": self.classes[pred_idx],
            "scores": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
        }

    def _to_tensor(self, image: np.ndarray) -> Any:
        import torch

        resized = resize_rgb_image(image, self.image_size).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized - mean) / std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor


def load_deep_inference(model_path: str | Path, class_names_path: str | Path) -> DeepInference:
    import torch
    from torch import nn
    from torchvision import models

    classes = json.loads(Path(class_names_path).read_text(encoding="utf-8"))
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))
    state = torch.load(Path(model_path), map_location="cpu")
    model.load_state_dict(state)
    return DeepInference(model=model, classes=classes)
