from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = Path(__file__).with_name("penguins_model.joblib")


def load_model():
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Serialized model not found: {MODEL_PATH}")
	return joblib.load(MODEL_PATH)


MODEL = load_model()


def _get_preprocessor(model):
	if "preprocessor" in model.named_steps:
		return model.named_steps["preprocessor"]
	if "pre" in model.named_steps:
		return model.named_steps["pre"]
	raise KeyError("The loaded pipeline does not contain a preprocessing step.")


REQUIRED_FEATURES = list(_get_preprocessor(MODEL).feature_names_in_)

app = Flask(__name__)


def _error(message, status_code=400):
	return jsonify({"error": message}), status_code


def _validate_record(record):
	if not isinstance(record, dict):
		return None, "Each instance must be a JSON object."

	missing = [column for column in REQUIRED_FEATURES if column not in record]
	if missing:
		return None, f"Missing required fields: {', '.join(missing)}"

	cleaned = {}
	for column in REQUIRED_FEATURES:
		value = record[column]
		if value is None:
			return None, f"Field '{column}' cannot be null."
		cleaned[column] = value

	return cleaned, None


@app.get("/health")
def health():
	return jsonify({"status": "ok", "model_loaded": True, "classes": MODEL.classes_.tolist()})


@app.post("/predict")
def predict():
	payload = request.get_json(silent=True)
	if payload is None:
		return _error("Request body must be valid JSON.")

	if isinstance(payload, dict) and "instances" in payload:
		records = payload["instances"]
	elif isinstance(payload, list):
		records = payload
	else:
		records = [payload]

	if not isinstance(records, list) or not records:
		return _error("Provide a single penguin record or a non-empty list of records.")

	cleaned_records = []
	for record in records:
		cleaned, error = _validate_record(record)
		if error:
			return _error(error)
		cleaned_records.append(cleaned)

	records_frame = pd.DataFrame(cleaned_records, columns=REQUIRED_FEATURES)
	predictions = MODEL.predict(records_frame)
	probabilities = MODEL.predict_proba(records_frame)

	response = []
	for predicted_species, probability_row in zip(predictions, probabilities):
		response.append(
			{
				"predicted_species": predicted_species,
				"probabilities": {
					class_name: float(probability)
					for class_name, probability in zip(MODEL.classes_, probability_row)
				},
			}
		)

	return jsonify({"predictions": response if len(response) > 1 else response[0]})


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
