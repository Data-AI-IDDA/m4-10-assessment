from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the serialized pipeline
try:
    model = joblib.load('penguin_model.joblib')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'API is healthy and running.'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # 1. Check if input data exists
    if not data:
        return jsonify({'error': 'No input data provided. Please provide JSON format.'}), 400
    
    # 2. Basic input validation
    required_features = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
    missing_features = [req for req in required_features if req not in data]
    
    if missing_features:
        return jsonify({'error': f'Missing required features: {missing_features}'}), 400
    
    try:
        # Convert JSON data to DataFrame (required for the Pipeline to work properly)
        df = pd.DataFrame([data])
        
        # Get prediction and probabilities
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0].tolist() # Convert to list for JSON serialization
        
        # Return the response in JSON format
        response = {
            'predicted_species': str(prediction),
            'probabilities': probabilities
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)
    