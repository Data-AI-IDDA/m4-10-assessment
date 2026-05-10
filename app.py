from flask import Flask, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
model = joblib.load("penguin_model.joblib")

@app.route("/health", methods=["GET"])
def health():
    return {"status": "healthy", "model_loaded": True}, 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    required = ["island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"]
    for f in required:
        if f not in data:
            return {"error": f"Missing field: '{f}'"}, 400
    data["sex"] = data["sex"].lower()
    df = pd.DataFrame([data])
    pred   = model.predict(df)[0]
    probs  = model.predict_proba(df)[0]
    names  = model.named_steps["classifier"].classes_
    return {"species": pred, "probabilities": dict(zip(names.tolist(), probs.tolist()))}, 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)