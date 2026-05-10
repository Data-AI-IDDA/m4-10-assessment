from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('penguin_model.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        input_df = pd.DataFrame([data])
        
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df).tolist()[0]
        classes = model.classes_.tolist()
        
        response = {
            "predicted_species": prediction,
            "probabilities": dict(zip(classes, probabilities))
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)