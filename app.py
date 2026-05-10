import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model once at startup
model = joblib.load("penguin_model.pkl")
REQUIRED_FIELDS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "island", "sex"]
VALID_ISLANDS = ["Torgersen", "Biscoe", "Dream"]
VALID_SEX = ["Male", "Female"]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "Penguin Species Classifier",
        "version": "1.0"
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # --- Input validation ---
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    if data["island"] not in VALID_ISLANDS:
        return jsonify({"error": f"Invalid island. Must be one of {VALID_ISLANDS}"}), 400

    if data["sex"] not in VALID_SEX:
        return jsonify({"error": f"Invalid sex. Must be one of {VALID_SEX}"}), 400

    for field in ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]:
        if not isinstance(data[field], (int, float)):
            return jsonify({"error": f"Field {field} must be a number"}), 400
        if data[field] <= 0:
            return jsonify({"error": f"Field {field} must be positive"}), 400

    # --- Prediction ---
    import pandas as pd
    input_df = pd.DataFrame([{
        "bill_length_mm":   data["bill_length_mm"],
        "bill_depth_mm":    data["bill_depth_mm"],
        "flipper_length_mm": data["flipper_length_mm"],
        "body_mass_g":      data["body_mass_g"],
        "island":           data["island"],
        "sex":              data["sex"]
    }])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_.tolist()

    return jsonify({
        "prediction": prediction,
        "probabilities": dict(zip(classes, probabilities.round(4).tolist()))
    }), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)