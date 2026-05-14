from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

MODEL_PATH = 'penguin_model_pipeline.joblib'
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

REQUIRED_FEATURES = [
    'island', 'bill_length_mm', 'bill_depth_mm', 
    'flipper_length_mm', 'body_mass_g', 'sex'
]

@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not available"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    missing_features = [feat for feat in REQUIRED_FEATURES if feat not in data]
    if missing_features:
        return jsonify({
            "error": "Missing features in input data",
            "missing": missing_features
        }), 400

    try:
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        class_probs = dict(zip(model.classes_, probabilities.tolist()))

        return jsonify({
            "predicted_species": prediction,
            "class_probabilities": class_probs,
            "model_used": "Logistic Regression (Tuned)"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)