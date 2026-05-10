import joblib
import pandas as pd
from flask import Flask, request, jsonify
import sys

app = Flask(__name__)

# ── Model loading with graceful error handling ──────────────────────────────
try:
    pipeline = joblib.load("penguin_pipeline.joblib")
except Exception as e:
    print(f"FATAL: Could not load model — {e}", file=sys.stderr)
    sys.exit(1)

REQUIRED_FIELDS = ["bill_length_mm", "bill_depth_mm",
                   "flipper_length_mm", "body_mass_g", "island", "sex"]
VALID_ISLANDS   = ["Biscoe", "Dream", "Torgersen"]
VALID_SEX       = ["Male", "Female"]
NUMERIC_FIELDS  = ["bill_length_mm", "bill_depth_mm",
                   "flipper_length_mm", "body_mass_g"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "penguin_pipeline.joblib"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON input provided"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    for field in NUMERIC_FIELDS:
        if not isinstance(data[field], (int, float)):
            return jsonify({"error": f"{field} must be a number"}), 400

    if data["island"] not in VALID_ISLANDS:
        return jsonify({"error": f"island must be one of {VALID_ISLANDS}"}), 400

    if data["sex"] not in VALID_SEX:
        return jsonify({"error": f"sex must be one of {VALID_SEX}"}), 400

    input_df = pd.DataFrame([{
        "bill_length_mm":    data["bill_length_mm"],
        "bill_depth_mm":     data["bill_depth_mm"],
        "flipper_length_mm": data["flipper_length_mm"],
        "body_mass_g":       data["body_mass_g"],
        "island":            data["island"],
        "sex":               data["sex"],
    }])

    prediction = pipeline.predict(input_df)[0]

    # predict_proba fallback: works for RF natively; wraps plain SVC gracefully
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)[0]
        classes       = pipeline.classes_
        prob_dict = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }
    else:
        prob_dict = {"note": "Model does not support probability estimates"}

    return jsonify({
        "predicted_species": prediction,
        "probabilities": prob_dict
    }), 200


if __name__ == "__main__":
    app.run(debug=False, port=5000)
