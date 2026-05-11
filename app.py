from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the full pipeline
model = joblib.load('penguin_model.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # 1. Basic Input Validation
    required_fields = [
        'island', 'bill_length_mm', 'bill_depth_mm', 
        'flipper_length_mm', 'body_mass_g', 'sex'
    ]
    
    if not data or not all(field in data for field in required_fields):
        return jsonify({
            "error": "Missing input data", 
            "required_fields": required_fields
        }), 400

    try:
        # 2. Convert JSON to DataFrame for the Pipeline
        input_df = pd.DataFrame([data])
        
        # 3. Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Map probabilities to class names
        prob_map = dict(zip(model.classes_, probabilities.tolist()))

        return jsonify({
            "species": prediction,
            "confidence_scores": prob_map
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)