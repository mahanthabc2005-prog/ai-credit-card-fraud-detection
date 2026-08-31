import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import current_app
from models.database_models import Transaction, db
from services.risk_service import classify_risk

FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


class PredictionError(Exception):
    pass


def validate_frame(frame):
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise PredictionError(f"Missing required columns: {', '.join(missing)}")
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise PredictionError("All transaction fields must be finite numeric values.")
    if (values["Amount"] < 0).any() or (values["Time"] < 0).any():
        raise PredictionError("Time and Amount cannot be negative.")
    return values


def _model():
    path = Path(current_app.config["MODEL_PATH"])
    if not path.exists():
        raise PredictionError("Model is unavailable. Run ml/train_model.py first.")
    return joblib.load(path)


def predict_frame(frame, save=True):
    clean = validate_frame(frame)
    model = _model()
    probabilities = model.predict_proba(clean)[:, 1]
    results = []
    for index, probability in enumerate(probabilities):
        risk = classify_risk(float(probability), current_app.config["RISK_LOW_THRESHOLD"], current_app.config["RISK_HIGH_THRESHOLD"])
        prediction = "Fraudulent" if probability >= 0.5 else "Legitimate"
        item = {
            "prediction": prediction, "fraud_probability": round(float(probability), 4),
            "risk_level": risk.level, "confidence": round(float(max(probability, 1 - probability)), 4),
            "recommendation": risk.recommendation,
            "explanation": "Risk estimate produced by the trained model; it is not proof of fraud.",
        }
        if save:
            record = Transaction(transaction_time=float(clean.iloc[index]["Time"]), amount=float(clean.iloc[index]["Amount"]), prediction=prediction, fraud_probability=float(probability), risk_level=risk.level)
            db.session.add(record)
            item["id"] = record
        results.append(item)
    if save:
        db.session.commit()
        for item in results:
            item["id"] = item["id"].id
    return results


def load_metrics():
    path = Path(current_app.config["METRICS_PATH"])
    return json.loads(path.read_text()) if path.exists() else None
