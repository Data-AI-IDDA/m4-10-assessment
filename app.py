
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("penguin_species_model.joblib")

REQUIRED_FIELDS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island",
    "sex"
]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON payload provided"}), 400

    missing_fields = [
        field for field in REQUIRED_FIELDS
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        class_probs = {
            cls: float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }

        return jsonify({
            "prediction": prediction,
            "probabilities": class_probs
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
