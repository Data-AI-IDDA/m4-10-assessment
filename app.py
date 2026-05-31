from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "penguin_species_pipeline.joblib"

REQUIRED_FIELDS = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]

NUMERIC_FIELDS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

CATEGORICAL_FIELDS = [
    "island",
    "sex",
]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Make sure the .joblib file is in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Penguin species prediction API is running"
    }), 200


def validate_input(data):
    if not isinstance(data, dict):
        return False, "Input must be a JSON object."

    missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    for field in NUMERIC_FIELDS:
        try:
            float(data[field])
        except (ValueError, TypeError):
            return False, f"Field '{field}' must be numeric."

    for field in CATEGORICAL_FIELDS:
        if data[field] is None or str(data[field]).strip() == "":
            return False, f"Field '{field}' must be a non-empty string."

    return True, None


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({
                "error": error_message
            }), 400

        input_df = pd.DataFrame([{
            "island": data["island"],
            "bill_length_mm": float(data["bill_length_mm"]),
            "bill_depth_mm": float(data["bill_depth_mm"]),
            "flipper_length_mm": float(data["flipper_length_mm"]),
            "body_mass_g": float(data["body_mass_g"]),
            "sex": data["sex"],
        }])

        prediction = model.predict(input_df)[0]

        response = {
            "predicted_species": prediction
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            if hasattr(model, "classes_"):
                class_names = model.classes_
            elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("model"), "classes_"):
                class_names = model.named_steps["model"].classes_
            else:
                class_names = [f"class_{i}" for i in range(len(probabilities))]

            response["class_probabilities"] = {
                str(class_name): float(prob)
                for class_name, prob in zip(class_names, probabilities)
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": "Prediction failed.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)