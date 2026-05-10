from flask import Flask, request, jsonify

import pandas as pd
import joblib

# Flask app
app = Flask(__name__)

# Load trained pipeline
model = joblib.load(
    r"C:\Users\User\Documents\IRONHACK\m4-10-assessment\penguin_model.joblib"
)
# Required input columns
required_columns = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island",
    "sex"
]

# Health endpoint
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok"
    })


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get JSON input
        data = request.get_json()

        # Validate required fields
        for col in required_columns:

            if col not in data:

                return jsonify({
                    "error": f"Missing column: {col}"
                }), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Predict species
        prediction = model.predict(input_df)[0]

        # Predict probabilities
        probabilities = model.predict_proba(input_df)[0]

        probability_dict = dict(
            zip(
                model.classes_,
                probabilities.tolist()
            )
        )

        # Return response
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
    
