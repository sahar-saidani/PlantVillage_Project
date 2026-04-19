from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(ROOT / "src"))

from plant_disease.inference import (
    build_visual_pipeline,
    load_classical_inference,
    load_deep_inference,
    resize_rgb_image,
)


st.set_page_config(page_title="Plant Disease Comparator", layout="wide")


def pil_to_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def show_scores(scores: dict[str, float] | None, title: str) -> None:
    st.subheader(title)
    if not scores:
        st.write("No probability scores available for this model.")
        return
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for label, score in ordered:
        st.write(f"{label}: {score:.4f}")


@st.cache_resource
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
        except Exception as exc:
            st.warning(f"Deep model unavailable: {exc}")
            deep = None

    return classical, deep


def load_comparison_summary() -> dict | None:
    path = ROOT / "artifacts" / "classical_ml" / "comparison_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


st.title("Plant Disease Detection Interface")
st.write(
    "Upload an image or use the camera. The app compares the classical ML model and the deep pretrained model."
)

classical_model, deep_model = load_models()
comparison_summary = load_comparison_summary()

source = st.radio("Image source", ["Upload image", "Camera"], horizontal=True)
uploaded = None
if source == "Upload image":
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
else:
    uploaded = st.camera_input("Take a photo")

if uploaded is not None:
    pil_image = Image.open(uploaded)
    image = pil_to_rgb(pil_image)
    image = resize_rgb_image(image, 224)

    visuals = build_visual_pipeline(image)

    st.subheader("Visual Pipeline")
    cols = st.columns(3)
    cols[0].image(visuals["original"], caption="Original", use_container_width=True)
    cols[1].image(visuals["preprocessed"], caption="Preprocessed", use_container_width=True)
    cols[2].image(visuals["advanced_preprocessed"], caption="Advanced preprocessing", use_container_width=True)

    cols = st.columns(4)
    cols[0].image(visuals["grayscale"], caption="Grayscale", use_container_width=True)
    cols[1].image(visuals["threshold_mask"], caption="Threshold", use_container_width=True)
    cols[2].image(visuals["hsv_mask"], caption="HSV mask", use_container_width=True)
    cols[3].image(visuals["kmeans_mask"], caption="KMeans mask", use_container_width=True)

    cols = st.columns(2)
    cols[0].image(visuals["advanced_mask"], caption="Advanced mask", use_container_width=True)
    cols[1].image(visuals["advanced_segmented_leaf"], caption="Advanced segmented leaf", use_container_width=True)

    cols = st.columns(3)
    cols[0].image(visuals["segmented_leaf"], caption="HSV segmented leaf", use_container_width=True)
    cols[1].image(visuals["sobel_edges"], caption="Sobel", use_container_width=True)
    cols[2].image(visuals["canny_edges"], caption="Canny", use_container_width=True)

    result_cols = st.columns(2)
    classical_result = None
    deep_result = None

    if classical_model is not None:
        classical_result = classical_model.predict(image)
        result_cols[0].metric("Classical ML prediction", classical_result["label"])
        show_scores(classical_result["scores"], "Classical ML scores")
    else:
        result_cols[0].warning("Classical model not found.")

    if deep_model is not None:
        deep_result = deep_model.predict(image)
        result_cols[1].metric("Deep model prediction", deep_result["label"])
        show_scores(deep_result["scores"], "Deep model scores")
    else:
        result_cols[1].warning("Deep model not found or dependencies missing.")

    st.subheader("Model Comparison")
    rows = []
    if classical_result is not None:
        rows.append({"model": "Classical ML", "prediction": classical_result["label"]})
    if deep_result is not None:
        rows.append({"model": "Deep Learning", "prediction": deep_result["label"]})
    if rows:
        st.table(rows)

if comparison_summary:
    st.subheader("Project-Level Comparison")
    st.json(comparison_summary)
else:
    st.info(
        "Run `python scripts/build_comparison_report.py` to generate a project-level comparison summary."
    )
