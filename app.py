
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the serialized pipeline at startup
model = joblib.load("penguin_pipeline.joblib")

# Feature definitions
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_version": "1.0.0"})

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Parse JSON
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # 2. Validate all required fields
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # 3. Validate types
    for feat in NUMERIC_FEATURES:
        if not isinstance(data[feat], (int, float)):
            return jsonify({"error": f"Field '{feat}' must be a number"}), 400
    for feat in CATEGORICAL_FEATURES:
        if not isinstance(data[feat], str):
            return jsonify({"error": f"Field '{feat}' must be a string"}), 400

    # 4. Predict (One single Try/Except block)
    try:
        input_df = pd.DataFrame([data])[ALL_FEATURES]

        # Convert NumPy types to Python types for JSON compatibility
        raw_prediction = model.predict(input_df)[0]
        prediction = raw_prediction.item() if hasattr(raw_prediction, 'item') else int(raw_prediction)

        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        return jsonify({
            "prediction": prediction,
            "probabilities": {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(classes, probabilities)
            },
            "model_version": "1.0.0"
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
