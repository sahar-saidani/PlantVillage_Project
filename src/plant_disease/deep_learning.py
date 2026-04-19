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
from .preprocessing import preprocess_image, read_image
from .utils import ensure_dir, read_csv


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
        image = preprocess_image(image)

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
            if np.random.rand() < 0.2:
                image = cv2.GaussianBlur(image, (3, 3), 0)

        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
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


def _split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


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

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            predictions = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(targets.tolist())

    metrics = classification_report_dict(y_true, y_pred, labels)
    metrics["history"] = history
    metrics["model_name"] = model_name

    ensure_dir(config.paths.deep_dir)
    torch.save(model.state_dict(), config.paths.deep_dir / "best_model.pt")
    save_metrics(config.paths.deep_dir, "deep_metrics.json", metrics)
    save_confusion_matrix_image(
        metrics["confusion_matrix"], labels, config.paths.deep_dir / "deep_confusion_matrix.png"
    )
    return metrics

