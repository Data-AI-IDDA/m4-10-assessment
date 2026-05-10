from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("best_penguin_model.joblib")

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Required input fields
    required_fields = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "island",
        "sex"
    ]

    # Validate input
    for field in required_fields:
        if field not in data:
            return jsonify({
                "error": f"Missing field: {field}"
            }), 400

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(df)[0]

        # Get class names
        class_names = model.classes_

        probability_dict = {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }

        return jsonify({
            "prediction": prediction,
            "probabilities": probability_dict
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# Run app
if __name__ == "__main__":
    app.run(debug=True)