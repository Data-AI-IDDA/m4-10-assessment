from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load pipeline
model = joblib.load("penguin_model_pipeline.pkl")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate required fields
        required_features = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "island",
            "sex"
        ]

        missing = [f for f in required_features if f not in data]

        if missing:
            return jsonify({
                "error": "Missing features",
                "missing": missing
            }), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        return jsonify({
            "prediction": str(prediction),
            "probabilities": probabilities.tolist()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)