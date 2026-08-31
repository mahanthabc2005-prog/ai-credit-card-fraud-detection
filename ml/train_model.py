"""Train a fraud-risk model from Credit Card Fraud CSV or synthetic demo data."""
import json
from pathlib import Path
import shutil
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def demo_data(rows=8000):
    x, y = make_classification(n_samples=rows, n_features=28, n_informative=12, n_redundant=6,
                               weights=[0.97, 0.03], flip_y=0.01, class_sep=1.2, random_state=42)
    frame = pd.DataFrame(x, columns=[f"V{i}" for i in range(1, 29)])
    rng = np.random.default_rng(42)
    # A reproducible academic demo pattern: fraud cases combine unusually large
    # amounts with several atypical anonymised feature values. A real dataset
    # should always replace this fallback before any real-world interpretation.
    frame.loc[y == 1, "V1"] -= 5.0
    frame.loc[y == 1, "V2"] += 4.5
    frame.loc[y == 1, "V3"] -= 4.0
    frame.loc[y == 1, "V4"] += 4.0
    frame.loc[y == 1, "V7"] += 4.5
    frame.loc[y == 1, "V10"] -= 4.5
    amount = rng.lognormal(3.8, 0.75, rows)
    amount[y == 1] = rng.lognormal(7.5, 0.35, int(y.sum()))
    frame.insert(0, "Amount", np.clip(amount, 1, 5000))
    frame.insert(0, "Time", rng.uniform(0, 172800, rows))
    frame["Class"] = y
    return frame


def load_data(path=None):
    if path and Path(path).exists():
        frame = pd.read_csv(path)
        required = set(FEATURES + ["Class"])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {sorted(missing)}")
        frame = frame.drop_duplicates().dropna(subset=FEATURES + ["Class"])
        return frame
    print("No dataset found; training with synthetic academic demo data.")
    return demo_data()


def train(dataset_path=None):
    RESULTS.mkdir(exist_ok=True)
    data = load_data(dataset_path)
    x, y = data[FEATURES], data["Class"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)
    prep = ColumnTransformer([("scale", StandardScaler(), FEATURES)])
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=16, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    trained, metrics = {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess", prep), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics[name] = {"precision": round(precision_score(y_test, prediction, zero_division=0), 4), "recall": round(recall_score(y_test, prediction, zero_division=0), 4), "f1": round(f1_score(y_test, prediction, zero_division=0), 4), "roc_auc": round(roc_auc_score(y_test, probability), 4), "pr_auc": round(average_precision_score(y_test, probability), 4)}
        trained[name] = pipeline
    selected = max(metrics, key=lambda key: (metrics[key]["f1"], metrics[key]["pr_auc"]))
    best, probability = trained[selected], trained[selected].predict_proba(x_test)[:, 1]
    joblib.dump(best, ROOT / "fraud_pipeline.joblib")
    metrics["selected_model"] = selected
    metrics["dataset_rows"] = int(len(data))
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cm = confusion_matrix(y_test, (probability >= 0.5).astype(int))
    plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix"); plt.xticks([0,1],["Legitimate","Fraud"]); plt.yticks([0,1],["Legitimate","Fraud"])
    for i in range(2):
        for j in range(2): plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(RESULTS / "confusion_matrix.png", dpi=160); plt.close()
    RocCurveDisplay.from_predictions(y_test, probability); plt.tight_layout(); plt.savefig(RESULTS / "roc_curve.png", dpi=160); plt.close()
    PrecisionRecallDisplay.from_predictions(y_test, probability); plt.tight_layout(); plt.savefig(RESULTS / "precision_recall_curve.png", dpi=160); plt.close()
    static_images = ROOT.parent / "static" / "images"
    static_images.mkdir(exist_ok=True)
    for image in ("confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"):
        shutil.copy2(RESULTS / image, static_images / image)
    return selected, metrics


if __name__ == "__main__":
    import sys
    selected, _ = train(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Saved {selected} pipeline to {ROOT / 'fraud_pipeline.joblib'}")
