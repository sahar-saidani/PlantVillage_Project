from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.inference import (  # noqa: E402
    build_visual_pipeline,
    load_classical_inference,
    load_deep_inference,
    resize_rgb_image,
)


app = Flask(__name__)


def pil_to_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def image_to_base64(image: np.ndarray) -> str:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Unable to encode image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def load_models() -> tuple[object | None, object | None]:
    classical = None
    deep = None

    classical_candidates = [
        ROOT / "artifacts" / "classical_ml" / "best_classical_model_clean.pkl",
        ROOT / "artifacts" / "classical_ml" / "best_classical_model.pkl",
    ]
    for candidate in classical_candidates:
        if candidate.exists():
            classical = load_classical_inference(candidate)
            break

    deep_model_path = ROOT / "artifacts" / "deep_learning" / "best_model.pt"
    deep_classes_path = ROOT / "artifacts" / "deep_learning" / "class_names.json"
    if deep_model_path.exists() and deep_classes_path.exists():
        try:
            deep = load_deep_inference(deep_model_path, deep_classes_path)
        except Exception:
            deep = None

    return classical, deep


CLASSICAL_MODEL, DEEP_MODEL = load_models()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "classical_model_loaded": CLASSICAL_MODEL is not None,
            "deep_model_loaded": DEEP_MODEL is not None,
        }
    )


@app.route("/api/comparison-summary", methods=["GET"])
def comparison_summary():
    path = ROOT / "artifacts" / "classical_ml" / "comparison_summary.json"
    if not path.exists():
        return jsonify({"error": "comparison_summary.json not found"}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "Missing image file in form-data under key 'image'"}), 400

    try:
        pil_image = Image.open(BytesIO(file.read()))
        image = resize_rgb_image(pil_to_rgb(pil_image), 224)
        visuals = build_visual_pipeline(image)
    except Exception as exc:
        return jsonify({"error": f"Unable to process image: {exc}"}), 400

    classical_result = CLASSICAL_MODEL.predict(image) if CLASSICAL_MODEL is not None else None
    deep_result = DEEP_MODEL.predict(image) if DEEP_MODEL is not None else None

    return jsonify(
        {
            "predictions": {
                "classical_ml": classical_result,
                "deep_learning": deep_result,
            },
            "visuals": {name: image_to_base64(output) for name, output in visuals.items()},
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
