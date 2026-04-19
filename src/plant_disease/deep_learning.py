from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import AppConfig
from .evaluation import classification_report_dict, save_confusion_matrix_image, save_metrics
from .preprocessing import read_image
from .utils import ensure_dir, read_csv, write_json


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantDiseaseDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_size: int,
        label_to_idx: dict[str, int],
        augment: bool = False,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.label_to_idx = label_to_idx
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = read_image(row["image_path"], self.image_size)

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
            if np.random.rand() < 0.5:
                angle = float(np.random.uniform(-15.0, 15.0))
                center = (self.image_size / 2, self.image_size / 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(
                    image,
                    matrix,
                    (self.image_size, self.image_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            if np.random.rand() < 0.5:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 1] *= np.random.uniform(0.85, 1.15)
                hsv[..., 2] *= np.random.uniform(0.85, 1.15)
                hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return tensor, self.label_to_idx[row["label"]]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _build_model(config: AppConfig, num_classes: int) -> tuple[nn.Module, str]:
    if config.deep_learning.use_pretrained_if_available:
        try:
            from torchvision import models

            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model, "efficientnet_b0_pretrained"
        except Exception:
            pass
    return SimpleCNN(num_classes), "simple_cnn"


def _build_model_from_state_dict(
    config: AppConfig,
    num_classes: int,
    state_dict: dict[str, torch.Tensor],
) -> tuple[nn.Module, str]:
    if any(key.startswith("features.0.0.") for key in state_dict):
        try:
            from torchvision import models

            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model, "efficientnet_b0_pretrained"
        except Exception as exc:
            raise RuntimeError(
                "The saved checkpoint appears to be an EfficientNet model, "
                "but torchvision is unavailable to rebuild the architecture."
            ) from exc
    return _build_model(config, num_classes)


def _split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def _build_test_loader(
    config: AppConfig,
    rows: list[dict[str, str]],
    labels: list[str],
) -> DataLoader:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    test_rows = _split_rows(rows, "test")
    test_ds = PlantDiseaseDataset(test_rows, config.dataset.image_size, label_to_idx, augment=False)
    return DataLoader(
        test_ds,
        batch_size=config.deep_learning.batch_size,
        shuffle=False,
        num_workers=config.deep_learning.num_workers,
    )


def _evaluate_model_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    labels: list[str],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            predictions = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(targets.tolist())
    return classification_report_dict(y_true, y_pred, labels)


def _write_deep_outputs(
    config: AppConfig,
    labels: list[str],
    metrics: dict[str, object],
    model_state: dict[str, torch.Tensor] | None = None,
) -> None:
    ensure_dir(config.paths.deep_dir)
    if model_state is not None:
        torch.save(model_state, config.paths.deep_dir / "best_model.pt")
    write_json(config.paths.deep_dir / "class_names.json", labels)
    save_metrics(config.paths.deep_dir, "deep_metrics.json", metrics)
    save_confusion_matrix_image(
        metrics["confusion_matrix"],
        labels,
        config.paths.deep_dir / "deep_confusion_matrix.png",
    )


def load_saved_deep_results(config: AppConfig) -> dict[str, object]:
    training_summary_path = config.paths.deep_dir / "training_summary.json"
    deep_metrics_path = config.paths.deep_dir / "deep_metrics.json"

    if training_summary_path.exists():
        import json

        payload = json.loads(training_summary_path.read_text(encoding="utf-8"))
        return {
            "accuracy": payload.get("test_accuracy"),
            "macro_f1": payload.get("macro_avg_f1"),
            "best_val_acc": payload.get("best_val_acc"),
            "history": payload.get("history", []),
            "model_name": payload.get("model_name", "saved_checkpoint"),
            "reused_pretrained_checkpoint": True,
            "loaded_from_summary_only": True,
        }

    if deep_metrics_path.exists():
        import json

        payload = json.loads(deep_metrics_path.read_text(encoding="utf-8"))
        payload["reused_pretrained_checkpoint"] = True
        payload["loaded_from_summary_only"] = True
        return payload

    raise FileNotFoundError(
        "No reusable deep artifacts found. Expected training_summary.json or deep_metrics.json."
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += int(inputs.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


def train_and_evaluate_deep(config: AppConfig, metadata_path: Path) -> dict[str, object]:
    rows = read_csv(metadata_path)
    labels = config.dataset.classes
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    train_rows = _split_rows(rows, "train")
    val_rows = _split_rows(rows, "val")
    test_rows = _split_rows(rows, "test")

    train_ds = PlantDiseaseDataset(train_rows, config.dataset.image_size, label_to_idx, augment=True)
    val_ds = PlantDiseaseDataset(val_rows, config.dataset.image_size, label_to_idx, augment=False)
    test_ds = PlantDiseaseDataset(test_rows, config.dataset.image_size, label_to_idx, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.deep_learning.batch_size,
        shuffle=True,
        num_workers=config.deep_learning.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.deep_learning.batch_size,
        shuffle=False,
        num_workers=config.deep_learning.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.deep_learning.batch_size,
        shuffle=False,
        num_workers=config.deep_learning.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = _build_model(config, len(labels))
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.deep_learning.lr,
        weight_decay=config.deep_learning.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state = None
    for epoch in range(config.deep_learning.epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = _evaluate_model_on_test(model, test_loader, labels, device)
    metrics["history"] = history
    metrics["model_name"] = model_name

    _write_deep_outputs(
        config,
        labels,
        metrics,
        model_state=model.state_dict(),
    )
    return metrics


def evaluate_saved_deep_model(config: AppConfig, metadata_path: Path) -> dict[str, object]:
    rows = read_csv(metadata_path)
    labels = config.dataset.classes
    classes_path = config.paths.deep_dir / "class_names.json"
    if classes_path.exists():
        import json

        saved_labels = json.loads(classes_path.read_text(encoding="utf-8"))
        if list(saved_labels) == labels:
            labels = saved_labels

    test_loader = _build_test_loader(config, rows, labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = config.paths.deep_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved deep model not found: {model_path}")

    state = torch.load(model_path, map_location=device)
    model, model_name = _build_model_from_state_dict(config, len(labels), state)
    model.load_state_dict(state)
    model = model.to(device)

    metrics = _evaluate_model_on_test(model, test_loader, labels, device)
    metrics["history"] = []
    metrics["model_name"] = f"{model_name}_reused"
    metrics["reused_pretrained_checkpoint"] = True

    _write_deep_outputs(config, labels, metrics)
    return metrics


def reuse_or_evaluate_saved_deep_model(config: AppConfig, metadata_path: Path) -> dict[str, object]:
    try:
        return evaluate_saved_deep_model(config, metadata_path)
    except Exception:
        return load_saved_deep_results(config)
