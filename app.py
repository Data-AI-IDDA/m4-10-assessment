from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('penguin_model.joblib')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_fields = ['island', 'bill_length_mm', 'bill_depth_mm', 
                           'flipper_length_mm', 'body_mass_g', 'sex']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_

        response = {
            "prediction": prediction,
            "probabilities": {cls: round(prob, 4) for cls, prob in zip(classes, probs)}
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
