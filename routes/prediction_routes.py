from io import StringIO
import pandas as pd
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from services.prediction_service import FEATURES, PredictionError, predict_frame

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")

@prediction_bp.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict): return jsonify(error="Send a JSON transaction object."), 400
        result = predict_frame(pd.DataFrame([payload]))[0]
        return jsonify(result), 201
    except PredictionError as error: return jsonify(error=str(error)), 400
    except Exception:
        return jsonify(error="Prediction could not be completed."), 500

@prediction_bp.post("/predict/batch")
def predict_batch():
    try:
        upload = request.files.get("file")
        if not upload or not upload.filename: return jsonify(error="Choose a CSV file."), 400
        if "." not in upload.filename or secure_filename(upload.filename).rsplit(".", 1)[1].lower() != "csv": return jsonify(error="Only CSV files are accepted."), 400
        frame = pd.read_csv(StringIO(upload.stream.read().decode("utf-8-sig")))
        if frame.empty: return jsonify(error="The CSV is empty."), 400
        results = predict_frame(frame)
        # Class is optional ground truth for academic test datasets. It is never
        # used by the model, so the returned prediction remains independent.
        if "Class" in frame.columns:
            labels = pd.to_numeric(frame["Class"], errors="coerce")
            if labels.isna().any() or not labels.isin([0, 1]).all():
                return jsonify(error="Optional Class values must be 0 or 1."), 400
            for item, label in zip(results, labels):
                item["ground_truth"] = "Fraudulent" if label == 1 else "Legitimate"
        return jsonify(results=results, summary={"transactions": len(results), "fraudulent": sum(x["prediction"] == "Fraudulent" for x in results), "high_risk": sum(x["risk_level"] == "HIGH" for x in results)}), 201
    except UnicodeDecodeError: return jsonify(error="CSV must be UTF-8 encoded."), 400
    except (pd.errors.ParserError, PredictionError) as error: return jsonify(error=str(error)), 400
    except Exception: return jsonify(error="Batch prediction could not be completed."), 500

@prediction_bp.get("/health")
def health(): return jsonify(status="ok")
