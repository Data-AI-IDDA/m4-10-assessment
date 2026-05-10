"""Flask deployment prototype for the Palmer Penguins classifier.

Run:
    python app.py

Endpoints:
    GET  /health   -> liveness + model-loaded flag
    POST /predict  -> predict species + class probabilities
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "penguin_pipeline.joblib"))

NUMERIC_FIELDS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FIELDS = ["island", "sex"]
REQUIRED_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS

app = Flask(__name__)

try:
    pipeline = joblib.load(MODEL_PATH)
    load_error: str | None = None
except Exception as exc:
    pipeline = None
    load_error = f"{type(exc).__name__}: {exc}"


def _validate(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "Request body must be a JSON object."
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return f"Missing fields: {missing}"
    for f in NUMERIC_FIELDS:
        try:
            float(payload[f])
        except (TypeError, ValueError):
            return f"Field '{f}' must be numeric."
    for f in CATEGORICAL_FIELDS:
        if not isinstance(payload[f], str) or not payload[f].strip():
            return f"Field '{f}' must be a non-empty string."
    return None


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "penguin-classifier",
        "endpoints": {
            "GET /health":  "liveness + model-loaded flag",
            "POST /predict": "predict species; body must be JSON with "
                             f"{REQUIRED_FIELDS}",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if pipeline is not None else "model_not_loaded",
        "model_loaded": pipeline is not None,
        "model_path": str(MODEL_PATH),
        "load_error": load_error,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model not loaded.", "detail": load_error}), 503

    payload = request.get_json(silent=True)
    error = _validate(payload)
    if error is not None:
        return jsonify({"error": error}), 400

    row = {f: [float(payload[f])] for f in NUMERIC_FIELDS}
    row.update({f: [payload[f]] for f in CATEGORICAL_FIELDS})
    frame = pd.DataFrame(row)

    try:
        prediction = pipeline.predict(frame)[0]
        proba = pipeline.predict_proba(frame)[0]
    except Exception as exc:
        return jsonify({"error": "Inference failed.", "detail": f"{type(exc).__name__}: {exc}"}), 500

    classes = list(pipeline.classes_)
    return jsonify({
        "prediction": str(prediction),
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
