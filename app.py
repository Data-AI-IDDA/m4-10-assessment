"""
app.py — Penguin Species Prediction API
Flask REST API that serves a trained sklearn pipeline for penguin classification.

Endpoints:
  GET  /health   — health check
  POST /predict  — predict species + class probabilities
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "penguin_pipeline.joblib")

REQUIRED_FIELDS = {
    "bill_length_mm":    float,
    "bill_depth_mm":     float,
    "flipper_length_mm": float,
    "body_mass_g":       float,
    "island":            str,
    "sex":               str,
}

VALID_ISLANDS = {"Torgersen", "Biscoe", "Dream"}
VALID_SEXES   = {"Male", "Female"}

# ---------------------------------------------------------------------------
# App & model initialization
# ---------------------------------------------------------------------------
app = Flask(__name__)

_model = None
_model_loaded = False

def load_model():
    global _model, _model_loaded
    if not _model_loaded:
        try:
            _model = joblib.load(MODEL_PATH)
            _model_loaded = True
            app.logger.info(f"Model loaded from {MODEL_PATH}")
        except FileNotFoundError:
            app.logger.error(f"Model file not found: {MODEL_PATH}")
        except Exception as e:
            app.logger.error(f"Failed to load model: {e}")

load_model()

# ---------------------------------------------------------------------------
# Helper — input validation
# ---------------------------------------------------------------------------
def validate_input(data: dict):
    """
    Returns (cleaned_dict, error_message).
    error_message is None if validation passes.
    """
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    # Check for missing fields
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return None, f"Missing required fields: {', '.join(missing)}"

    cleaned = {}

    # Type coercion & range checks for numeric fields
    numeric_fields = {k: v for k, v in REQUIRED_FIELDS.items() if v is float}
    for field in numeric_fields:
        try:
            val = float(data[field])
        except (TypeError, ValueError):
            return None, f"Field '{field}' must be a number, got: {repr(data[field])}"

        # Sanity ranges (generous bounds for penguin measurements)
        bounds = {
            "bill_length_mm":    (10.0,  80.0),
            "bill_depth_mm":     (5.0,   30.0),
            "flipper_length_mm": (100.0, 280.0),
            "body_mass_g":       (500.0, 8000.0),
        }
        lo, hi = bounds[field]
        if not (lo <= val <= hi):
            return None, (
                f"Field '{field}' value {val} is outside expected range [{lo}, {hi}]."
            )
        cleaned[field] = val

    # Categorical validation
    island = str(data["island"]).strip().capitalize()
    if island not in VALID_ISLANDS:
        return None, f"'island' must be one of {sorted(VALID_ISLANDS)}, got: {repr(data['island'])}"
    cleaned["island"] = island

    sex_raw = str(data["sex"]).strip()
    # Accept 'male'/'female' case-insensitively
    sex = sex_raw[0].upper() + sex_raw[1:].lower() if sex_raw else ""
    if sex not in VALID_SEXES:
        return None, f"'sex' must be one of {sorted(VALID_SEXES)}, got: {repr(data['sex'])}"
    cleaned["sex"] = sex

    return cleaned, None

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": _model_loaded}), 200


@app.post("/predict")
def predict():
    # --- Model availability check ---
    if not _model_loaded or _model is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 503

    # --- Parse JSON ---
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # --- Validate ---
    cleaned, error = validate_input(data)
    if error:
        return jsonify({"error": error}), 400

    # --- Predict ---
    try:
        input_df = pd.DataFrame([cleaned])
        pred_species    = _model.predict(input_df)[0]
        pred_proba      = _model.predict_proba(input_df)[0]
        class_labels    = _model.classes_.tolist()
        probabilities   = {cls: round(float(p), 4) for cls, p in zip(class_labels, pred_proba)}
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "predicted_species": pred_species,
        "probabilities":     probabilities,
    }), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
