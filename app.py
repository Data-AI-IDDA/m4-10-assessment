from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "penguin_species_pipeline.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

REQUIRED_FIELDS = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex"
]

VALID_ISLANDS = ["Torgersen", "Biscoe", "Dream"]
VALID_SEX_VALUES = ["Male", "Female"]


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

        missing_fields = [field for field in REQUIRED_FIELDS if field not in data]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400

        if data["island"] not in VALID_ISLANDS:
            return jsonify({
                "error": "Invalid island value.",
                "valid_values": VALID_ISLANDS
            }), 400

        if data["sex"] not in VALID_SEX_VALUES:
            return jsonify({
                "error": "Invalid sex value.",
                "valid_values": VALID_SEX_VALUES
            }), 400

        numeric_fields = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g"
        ]

        for field in numeric_fields:
            if not isinstance(data[field], (int, float)):
                return jsonify({
                    "error": f"Field '{field}' must be numeric."
                }), 400

        input_df = pd.DataFrame([{
            "island": data["island"],
            "bill_length_mm": data["bill_length_mm"],
            "bill_depth_mm": data["bill_depth_mm"],
            "flipper_length_mm": data["flipper_length_mm"],
            "body_mass_g": data["body_mass_g"],
            "sex": data["sex"]
        }])

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        probability_dict = {
            class_name: float(probability)
            for class_name, probability in zip(model.classes_, probabilities)
        }

        return jsonify({
            "predicted_species": prediction,
            "class_probabilities": probability_dict
        })

    except Exception as error:
        return jsonify({
            "error": "Prediction failed.",
            "details": str(error)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
