from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("penguin_pipeline.joblib")

# Required input fields
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
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Check if request body exists
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Validate required fields
    missing = [field for field in REQUIRED_FIELDS if field not in data]

    if missing:
        return jsonify({
            "error": f"Missing fields: {missing}"
        }), 400

    try:
        # Create dataframe
        sample = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(sample)[0]

        # Probabilities
        probabilities = model.predict_proba(sample)[0]

        prob_dict = {
            class_name: float(prob)
            for class_name, prob in zip(model.classes_, probabilities)
        }

        return jsonify({
            "predicted_species": prediction,
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)