from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
try:
    model = joblib.load('penguin_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Basic validation
        required_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island', 'sex']
        if not all(feature in data for feature in required_features):
            return jsonify({"error": f"Missing features. Required: {required_features}"}), 400
            
        # Convert to DataFrame (pipeline expects DF to handle column names correctly)
        df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0].tolist()
        classes = model.classes_.tolist()
        
        prob_dict = dict(zip(classes, probabilities))
        
        return jsonify({
            "predicted_species": prediction,
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)