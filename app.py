from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("penguin_model.pkl")


# -----------------------------------
# Health check endpoint
# -----------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# -----------------------------------
# Prediction endpoint
# -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    
    try:
        data = request.get_json()

        # Required features
        required_features = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "species",
            "island",
            "sex"
        ]

        # Validation
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        classes = model.classes_

        prob_dict = {
            cls: float(prob)
            for cls, prob in zip(classes, probabilities)
        }

        return jsonify({
            "predicted_species": prediction,
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True) 