from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_dir, write_json


def classification_report_dict(
    y_true: list[int], y_pred: list[int], labels: list[str]
) -> dict[str, object]:
    n_classes = len(labels)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        confusion[truth, pred] += 1

    per_class = {}
    for idx, label in enumerate(labels):
        tp = confusion[idx, idx]
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[idx, :].sum()),
        }

    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    macro_f1 = float(np.mean([metrics["f1"] for metrics in per_class.values()]))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def save_metrics(output_dir: Path, filename: str, metrics: dict[str, object]) -> None:
    ensure_dir(output_dir)
    write_json(output_dir / filename, metrics)


def save_confusion_matrix_image(
    matrix: list[list[int]], labels: list[str], output_path: Path
) -> None:
    ensure_dir(output_path.parent)
    matrix_np = np.asarray(matrix, dtype=np.float32)
    size = 100 + 100 * len(labels)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if matrix_np.max() > 0:
        norm = matrix_np / matrix_np.max()
    else:
        norm = matrix_np

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = int(255 - norm[i, j] * 180)
            x0, y0 = 100 + j * 100, 100 + i * 100
            canvas[y0 : y0 + 100, x0 : x0 + 100] = (value, value, 255)
            cv2.putText(
                canvas,
                str(int(matrix_np[i, j])),
                (x0 + 30, y0 + 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

    for idx, label in enumerate(labels):
        cv2.putText(
            canvas,
            label[:12],
            (100 + idx * 100, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label[:12],
            (10, 135 + idx * 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)

