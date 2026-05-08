import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("penguin_pipeline.pkl")

REQUIRED_FIELDS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island",
    "sex"
]

VALID_ISLANDS = ["Torgersen", "Biscoe", "Dream"]
VALID_SEX     = ["Male", "Female"]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # --- Validation ---
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    if data["island"] not in VALID_ISLANDS:
        return jsonify({"error": f"island must be one of {VALID_ISLANDS}"}), 400

    if data["sex"] not in VALID_SEX:
        return jsonify({"error": f"sex must be one of {VALID_SEX}"}), 400

    numeric_fields = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]
    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            return jsonify({"error": f"{field} must be numeric"}), 400
        if data[field] <= 0:
            return jsonify({"error": f"{field} must be positive"}), 400

    # --- Predict ---
    input_df = pd.DataFrame([{
        "bill_length_mm":   data["bill_length_mm"],
        "bill_depth_mm":    data["bill_depth_mm"],
        "flipper_length_mm":data["flipper_length_mm"],
        "body_mass_g":      data["body_mass_g"],
        "island":           data["island"],
        "sex":              data["sex"]
    }])

    prediction   = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes      = model.classes_.tolist()

    return jsonify({
        "prediction": prediction,
        "probabilities": dict(zip(classes, np.round(probabilities, 4).tolist()))
    }), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)