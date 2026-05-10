from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "penguin_model.joblib"

# Load model once when the app starts
model = joblib.load(MODEL_PATH)

REQUIRED_FEATURES = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex"
]

NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

CATEGORICAL_FEATURES = [
    "island",
    "sex"
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Penguin species prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Invalid request. JSON body is required."
            }), 400

        missing_features = [
            feature for feature in REQUIRED_FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features.",
                "missing_features": missing_features,
                "required_features": REQUIRED_FEATURES
            }), 400

        # Basic numeric validation
        for feature in NUMERIC_FEATURES:
            try:
                data[feature] = float(data[feature])
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Feature '{feature}' must be numeric."
                }), 400

        # Basic categorical validation
        for feature in CATEGORICAL_FEATURES:
            if not isinstance(data[feature], str):
                return jsonify({
                    "error": f"Feature '{feature}' must be a string."
                }), 400

        # Create a single-row DataFrame in the same format used during training
        input_df = pd.DataFrame([{
            "island": data["island"],
            "bill_length_mm": data["bill_length_mm"],
            "bill_depth_mm": data["bill_depth_mm"],
            "flipper_length_mm": data["flipper_length_mm"],
            "body_mass_g": data["body_mass_g"],
            "sex": data["sex"]
        }])

        predicted_species = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        probability_dict = {
            class_name: round(float(prob), 4)
            for class_name, prob in zip(model.classes_, probabilities)
        }

        return jsonify({
            "predicted_species": predicted_species,
            "class_probabilities": probability_dict
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)