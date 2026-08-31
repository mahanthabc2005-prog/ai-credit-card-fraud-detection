import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'fraud_detection.db'}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    MODEL_PATH = BASE_DIR / "ml" / "fraud_pipeline.joblib"
    METRICS_PATH = BASE_DIR / "ml" / "results" / "metrics.json"
    RISK_LOW_THRESHOLD = float(os.getenv("RISK_LOW_THRESHOLD", "0.30"))
    RISK_HIGH_THRESHOLD = float(os.getenv("RISK_HIGH_THRESHOLD", "0.70"))
    ALLOWED_EXTENSIONS = {"csv"}
