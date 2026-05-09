"""
app.py — Palmer Penguins Species Prediction API
================================================
Serves predictions from the trained RandomForest pipeline.

Endpoints
---------
GET  /health   → API liveness check + model info
POST /predict  → Predict species from penguin measurements
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ── Load model at startup ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "penguin_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    CLASSES = list(model.classes_)
    print(f"[startup] Model loaded from {MODEL_PATH}")
    print(f"[startup] Classes: {CLASSES}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Run the notebook first to serialise the model."
    )

# ── Required input fields ──────────────────────────────────────────────────
REQUIRED_FIELDS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island",
    "sex",
    "year",
]

NUMERIC_FIELDS  = {"bill_length_mm", "bill_depth_mm",
                   "flipper_length_mm", "body_mass_g", "year"}
VALID_ISLANDS   = {"Torgersen", "Biscoe", "Dream"}
VALID_SEX       = {"male", "female"}


def validate_input(data: dict) -> str | None:
    """Return an error message string, or None if input is valid."""
    # Missing fields
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"

    # Numeric type checks
    for field in NUMERIC_FIELDS:
        try:
            float(data[field])
        except (TypeError, ValueError):
            return f"Field '{field}' must be a number, got: {data[field]!r}"

    # Categorical value checks
    if data["island"] not in VALID_ISLANDS:
        return (
            f"Invalid island '{data['island']}'. "
            f"Must be one of: {sorted(VALID_ISLANDS)}"
        )
    if data["sex"] not in VALID_SEX:
        return (
            f"Invalid sex '{data['sex']}'. "
            f"Must be one of: {sorted(VALID_SEX)}"
        )

    return None  # all good


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness check — always returns 200 if the server is up."""
    return jsonify({
        "status": "ok",
        "model":   os.path.basename(MODEL_PATH),
        "classes": CLASSES,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict penguin species from measurements.

    Expected JSON body:
    {
        "bill_length_mm":    46.1,
        "bill_depth_mm":     13.2,
        "flipper_length_mm": 211.0,
        "body_mass_g":       4500.0,
        "island":            "Dream",
        "sex":               "male",
        "year":              2008
    }
    """
    # ── Parse JSON ──────────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # ── Validate ────────────────────────────────────────────────────────────
    error = validate_input(data)
    if error:
        return jsonify({"error": error}), 400

    # ── Build DataFrame (same column order the pipeline expects) ─────────
    row = pd.DataFrame([{
        "bill_length_mm":    float(data["bill_length_mm"]),
        "bill_depth_mm":     float(data["bill_depth_mm"]),
        "flipper_length_mm": float(data["flipper_length_mm"]),
        "body_mass_g":       float(data["body_mass_g"]),
        "year":              int(data["year"]),
        "island":            data["island"],
        "sex":               data["sex"],
    }])

    # ── Predict ─────────────────────────────────────────────────────────────
    try:
        prediction  = model.predict(row)[0]
        probabilities = dict(zip(CLASSES, model.predict_proba(row)[0].round(4)))
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500

    return jsonify({
        "prediction":    prediction,
        "probabilities": probabilities,
    }), 200


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
