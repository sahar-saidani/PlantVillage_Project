from __future__ import annotations

import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    classical_dir = root / "artifacts" / "classical_ml"
    deep_dir = root / "artifacts" / "deep_learning"
    output_dir = classical_dir

    classical_summary = None
    for candidate in [
        classical_dir / "classical_clean_summary.json",
        classical_dir / "classical_notebook_summary.json",
        classical_dir / "classical_notebook_summary_v2.json",
    ]:
        if candidate.exists():
            classical_summary = read_json(candidate)
            break

    if classical_summary is None:
        raise FileNotFoundError("No classical summary JSON found in artifacts/classical_ml")

    deep_summary = read_json(deep_dir / "training_summary.json")

    comparison = {
        "classes": deep_summary["selected_classes"],
        "classical": {
            "best_model": classical_summary["best_model_name"],
            "validation_results": classical_summary.get("validation_results", {}),
            "test_accuracy": classical_summary.get("test_accuracy"),
            "test_macro_f1": classical_summary.get("test_macro_f1"),
        },
        "deep_learning": {
            "model": "EfficientNet-B0 pretrained",
            "test_accuracy": deep_summary["test_accuracy"],
            "test_macro_f1": deep_summary["macro_avg_f1"],
            "best_val_acc": deep_summary["best_val_acc"],
        },
    }

    c_acc = comparison["classical"]["test_accuracy"]
    d_acc = comparison["deep_learning"]["test_accuracy"]
    if c_acc is not None:
        comparison["delta_accuracy_deep_minus_classical"] = d_acc - c_acc
        comparison["best_overall"] = (
            "deep_learning" if d_acc >= c_acc else "classical_ml"
        )
    else:
        comparison["delta_accuracy_deep_minus_classical"] = None
        comparison["best_overall"] = "deep_learning"

    out_path = output_dir / "comparison_summary.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Comparison report saved to {out_path}")


if __name__ == "__main__":
    main()

