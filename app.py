from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "penguin_model.joblib"
REQUIRED_FIELDS = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
NUMERIC_FIELDS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
VALID_ISLANDS = {"Torgersen", "Biscoe", "Dream"}
VALID_SEX = {"Male", "Female"}

try:
    model = joblib.load(MODEL_PATH)
    model_status = "loaded"
except FileNotFoundError:
    model = None
    model_status = "not loaded"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": model_status})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run the notebook first to generate penguin_model.joblib."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    for field in NUMERIC_FIELDS:
        try:
            val = float(data[field])
            if val <= 0:
                return jsonify({"error": f"{field} must be a positive number"}), 400
            data[field] = val
        except (ValueError, TypeError):
            return jsonify({"error": f"{field} must be numeric, got: {data[field]!r}"}), 400

    if data["island"] not in VALID_ISLANDS:
        return jsonify({"error": f"island must be one of {sorted(VALID_ISLANDS)}, got: {data['island']!r}"}), 400
    if data["sex"] not in VALID_SEX:
        return jsonify({"error": f"sex must be one of {sorted(VALID_SEX)}, got: {data['sex']!r}"}), 400

    input_df = pd.DataFrame([{
        "island":           data["island"],
        "bill_length_mm":   data["bill_length_mm"],
        "bill_depth_mm":    data["bill_depth_mm"],
        "flipper_length_mm": data["flipper_length_mm"],
        "body_mass_g":      data["body_mass_g"],
        "sex":              data["sex"],
    }])

    species = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return jsonify({
        "species": species,
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(model.classes_, probabilities)
        },
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
