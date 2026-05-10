"""
app.py — Penguin Species Prediction API
Task 4: Model Deployment Prototype

Start: python app.py
Endpoints:
  GET  /health   → health check
  POST /predict  → predict species from measurements
"""

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = 'penguin_model.joblib'

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
    print(f'[OK] Model loaded from {MODEL_PATH}')
except FileNotFoundError:
    model = None
    print(f'[WARN] Model not found at {MODEL_PATH}. Run Task 4 cell in the notebook first.')

# Required input fields and their expected types
REQUIRED_FIELDS = {
    'bill_length_mm':    float,
    'bill_depth_mm':     float,
    'flipper_length_mm': float,
    'body_mass_g':       float,
    'island':            str,
    'sex':               str,
}

VALID_ISLANDS = {'Torgersen', 'Biscoe', 'Dream'}
VALID_SEXES   = {'Male', 'Female'}
SPECIES       = ['Adelie', 'Chinstrap', 'Gentoo']


def validate_input(data: dict):
    """Validate and coerce incoming JSON. Returns (payload_df, error_str)."""
    missing = [k for k in REQUIRED_FIELDS if k not in data]
    if missing:
        return None, f'Missing required fields: {missing}'

    coerced = {}
    for field, dtype in REQUIRED_FIELDS.items():
        try:
            coerced[field] = dtype(data[field])
        except (ValueError, TypeError):
            return None, f'Field "{field}" must be of type {dtype.__name__}'

    if coerced['island'] not in VALID_ISLANDS:
        return None, f'island must be one of {sorted(VALID_ISLANDS)}'
    if coerced['sex'] not in VALID_SEXES:
        return None, f'sex must be one of {sorted(VALID_SEXES)}'

    for f in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
        if coerced[f] <= 0:
            return None, f'"{f}" must be a positive number'

    df = pd.DataFrame([coerced])
    return df, None


# ── /health ───────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok' if model is not None else 'degraded',
        'model':  MODEL_PATH,
        'loaded': model is not None
    })


# ── /predict ──────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 503

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    payload_df, error = validate_input(data)
    if error:
        return jsonify({'error': error}), 400

    try:
        species_pred = model.predict(payload_df)[0]
        proba        = model.predict_proba(payload_df)[0]
        proba_dict   = {cls: round(float(p), 4) for cls, p in zip(SPECIES, proba)}

        return jsonify({
            'species':       species_pred,
            'probabilities': proba_dict
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
