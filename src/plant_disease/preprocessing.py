from __future__ import annotations

import cv2
import numpy as np


def read_image(path: str, image_size: int) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def rgb_histograms(image: np.ndarray, bins: int = 32) -> list[np.ndarray]:
    histograms = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256]).flatten()
        histograms.append(hist / max(hist.sum(), 1.0))
    return histograms


def hsv_histograms(image: np.ndarray, bins: int = 32) -> list[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    histograms = []
    ranges = [(0, 180), (0, 256), (0, 256)]
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [bins], list(ranges[channel])).flatten()
        histograms.append(hist / max(hist.sum(), 1.0))
    return histograms


def leaf_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([20, 20, 20], dtype=np.uint8)
    upper_green = np.array([110, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def threshold_mask(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def kmeans_mask(image: np.ndarray, k: int = 3) -> np.ndarray:
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.uint8)
    green_scores = centers[:, 1].astype(np.int32) - (
        0.5 * centers[:, 0].astype(np.int32) + 0.5 * centers[:, 2].astype(np.int32)
    )
    leaf_cluster = int(np.argmax(green_scores))
    mask = (labels.flatten() == leaf_cluster).astype(np.uint8).reshape(image.shape[:2]) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def sobel_edges(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def edge_map(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    return cv2.Canny(gray, 80, 160)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


def contour_overlay(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)
    return overlay
