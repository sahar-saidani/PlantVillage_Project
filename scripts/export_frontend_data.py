from __future__ import annotations

import shutil
from pathlib import Path


def copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree_files(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    public_dir = root / "frontend" / "public" / "project-data"

    if public_dir.exists():
        shutil.rmtree(public_dir)
    public_dir.mkdir(parents=True, exist_ok=True)

    copy_if_exists(root / "artifacts" / "metadata" / "dataset_summary.json", public_dir / "dataset_summary.json")
    copy_if_exists(
        root / "artifacts" / "metadata" / "segmentation_benchmark.json",
        public_dir / "segmentation_benchmark.json",
    )
    copy_if_exists(
        root / "artifacts" / "classical_ml" / "comparison_summary.json",
        public_dir / "comparison_summary.json",
    )
    copy_if_exists(
        root / "artifacts" / "classical_ml" / "classical_clean_summary.json",
        public_dir / "classical_clean_summary.json",
    )
    copy_if_exists(
        root / "artifacts" / "classical_ml" / "svm_metrics.json",
        public_dir / "svm_metrics.json",
    )
    copy_if_exists(
        root / "artifacts" / "classical_ml" / "random_forest_metrics.json",
        public_dir / "random_forest_metrics.json",
    )
    copy_if_exists(
        root / "artifacts" / "deep_learning" / "training_summary.json",
        public_dir / "training_summary.json",
    )
    copy_if_exists(
        root / "artifacts" / "deep_learning" / "deep_metrics.json",
        public_dir / "deep_metrics.json",
    )
    copy_if_exists(
        root / "artifacts" / "classical_ml" / "svm_confusion_matrix.png",
        public_dir / "svm_confusion_matrix.png",
    )
    copy_if_exists(
        root / "artifacts" / "classical_ml" / "random_forest_confusion_matrix.png",
        public_dir / "random_forest_confusion_matrix.png",
    )
    copy_if_exists(
        root / "artifacts" / "deep_learning" / "deep_confusion_matrix.png",
        public_dir / "deep_confusion_matrix.png",
    )
    copy_if_exists(
        root / "artifacts" / "deep_learning" / "confusion_matrix.png",
        public_dir / "colab_confusion_matrix.png",
    )
    copy_tree_files(root / "artifacts" / "examples", public_dir / "examples")

    print(f"Exported frontend data to {public_dir}")


if __name__ == "__main__":
    main()
