from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load the trained model at startup
try:
    model = joblib.load('model.joblib')
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    if model is None:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 500
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json(force=True)
        
        # Validate required fields
        required_fields = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(df)[0]
        
        # Get probabilities
        probabilities = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            classes = model.classes_
            probabilities = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
        
        response = {
            "prediction": str(prediction)
        }
        if probabilities:
            response["probabilities"] = probabilities
            
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
