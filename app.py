import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

PIPELINE = joblib.load("penguin_pipeline.pkl")

REQUIRED_FIELDS = {
    "island":            str,
    "bill_length_mm":    float,
    "bill_depth_mm":     float,
    "flipper_length_mm": float,
    "body_mass_g":       float,
    "sex":               str,
}

VALID_ISLANDS = {"Biscoe", "Dream", "Torgersen"}
VALID_SEXES   = {"Male", "Female"}


def validate_input(data):
    for field, dtype in REQUIRED_FIELDS.items():
        if field not in data:
            return False, "Missing required field: " + field
        try:
            data[field] = dtype(data[field])
        except (ValueError, TypeError):
            return False, "Field " + field + " must be " + dtype.__name__

    if data["island"] not in VALID_ISLANDS:
        return False, "island must be one of: Biscoe, Dream, Torgersen"
    if data["sex"] not in VALID_SEXES:
        return False, "sex must be one of: Male, Female"
    if not (10 < data["bill_length_mm"] < 100):
        return False, "bill_length_mm out of plausible range (10-100 mm)"
    if not (5 < data["bill_depth_mm"] < 50):
        return False, "bill_depth_mm out of plausible range (5-50 mm)"
    if not (100 < data["flipper_length_mm"] < 300):
        return False, "flipper_length_mm out of plausible range (100-300 mm)"
    if not (500 < data["body_mass_g"] < 8000):
        return False, "body_mass_g out of plausible range (500-8000 g)"

    return True, ""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "model":   "Logistic Regression - Palmer Penguins",
        "version": "1.0.0",
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()

    is_valid, error_msg = validate_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 422

    X_input = pd.DataFrame([{
        "island":            data["island"],
        "bill_length_mm":    data["bill_length_mm"],
        "bill_depth_mm":     data["bill_depth_mm"],
        "flipper_length_mm": data["flipper_length_mm"],
        "body_mass_g":       data["body_mass_g"],
        "sex":               data["sex"],
    }])

    prediction    = PIPELINE.predict(X_input)[0]
    probabilities = PIPELINE.predict_proba(X_input)[0]
    classes       = PIPELINE.classes_.tolist()

    return jsonify({
        "prediction": prediction,
        "probabilities": {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        },
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)