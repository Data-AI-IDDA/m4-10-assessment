# Save this code as app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load the serialized model at startup
model = joblib.load('penguin_model.pkl')

# Define the exact features the model expects
EXPECTED_FEATURES = [
    'bill_length_mm', 'bill_depth_mm', 
    'flipper_length_mm', 'body_mass_g', 
    'island', 'sex'
]

# 2. Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': True
    }), 200

# 3. Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Basic Validation 1: Check if JSON exists
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400

    # Basic Validation 2: Check for missing features
    missing_features = [f for f in EXPECTED_FEATURES if f not in data]
    if missing_features:
        return jsonify({
            'error': 'Missing required features', 
            'missing': missing_features
        }), 400

    try:
        # Convert JSON dictionary to a pandas DataFrame 
        # (The pipeline requires a 2D structure with exact column names)
        input_df = pd.DataFrame([data])
        
        # Make predictions
        prediction = model.predict(input_df).item()
        probabilities = model.predict_proba(input_df).tolist()
        classes = model.classes_.tolist()

        # Zip classes and probabilities together into a readable dictionary
        prob_dict = dict(zip(classes, probabilities))

        # Return the results
        return jsonify({
            'prediction': prediction,
            'probabilities': prob_dict
        }), 200

    except Exception as e:
        # Catch any unexpected errors (e.g., wrong data types)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on local development server
    app.run(debug=True, host='0.0.0.0', port=5000)