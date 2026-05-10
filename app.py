from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the serialized pipeline
model = joblib.load('penguin_model.joblib')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Basic validation: check if required fields are present
        required_fields = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Convert input to DataFrame for the pipeline
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Get class labels
        classes = model.classes_
        prob_dict = {label: float(prob) for label, prob in zip(classes, probabilities)}
        
        return jsonify({
            "species": prediction,
            "probabilities": prob_dict
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
