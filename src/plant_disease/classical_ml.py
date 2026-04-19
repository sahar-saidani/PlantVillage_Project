from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .evaluation import classification_report_dict, save_confusion_matrix_image, save_metrics
from .features import extract_features
from .preprocessing import read_image
from .utils import ensure_dir, read_csv


def _require_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the classical ML pipeline. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return RandomForestClassifier, Pipeline, StandardScaler, SVC


def _load_split_features(
    config: AppConfig, rows: list[dict[str, str]], split: str, label_to_idx: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    x_data: list[np.ndarray] = []
    y_data: list[int] = []
    for row in rows:
        if row["split"] != split:
            continue
        image = read_image(row["image_path"], config.dataset.image_size)
        x_data.append(extract_features(image, config.classical_ml.hsv_bins))
        y_data.append(label_to_idx[row["label"]])
    return np.stack(x_data), np.asarray(y_data)


def train_and_evaluate_classical(config: AppConfig, metadata_path: Path) -> dict[str, object]:
    RandomForestClassifier, Pipeline, StandardScaler, SVC = _require_sklearn()

    ensure_dir(config.paths.classical_dir)
    rows = read_csv(metadata_path)
    labels = config.dataset.classes
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    x_train, y_train = _load_split_features(config, rows, "train", label_to_idx)
    x_test, y_test = _load_split_features(config, rows, "test", label_to_idx)

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(C=config.classical_ml.svm_c, kernel="rbf")),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=config.classical_ml.random_forest_trees,
        random_state=config.seed,
        n_jobs=-1,
    )

    models = {
        "svm": svm,
        "random_forest": forest,
    }

    all_results: dict[str, object] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = classification_report_dict(y_test.tolist(), predictions.tolist(), labels)
        save_metrics(config.paths.classical_dir, f"{name}_metrics.json", metrics)
        save_confusion_matrix_image(
            metrics["confusion_matrix"],
            labels,
            config.paths.classical_dir / f"{name}_confusion_matrix.png",
        )
        all_results[name] = metrics

    return all_results

