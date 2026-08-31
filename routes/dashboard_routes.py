from flask import Blueprint, jsonify, request
from sqlalchemy import func
from models.database_models import Transaction
from services.prediction_service import load_metrics

dashboard_bp = Blueprint("dashboard_api", __name__, url_prefix="/api")

@dashboard_bp.get("/dashboard")
def dashboard_data():
    total = Transaction.query.count()
    fraud = Transaction.query.filter_by(prediction="Fraudulent").count()
    high = Transaction.query.filter_by(risk_level="HIGH").count()
    average = Transaction.query.with_entities(func.avg(Transaction.amount)).scalar() or 0
    risks = {level: Transaction.query.filter_by(risk_level=level).count() for level in ("LOW", "MEDIUM", "HIGH")}
    recent = [item.to_dict() for item in Transaction.query.order_by(Transaction.created_at.desc()).limit(10)]
    return jsonify(total_transactions=total, fraudulent_transactions=fraud, legitimate_transactions=total-fraud, high_risk_transactions=high, fraud_percentage=round((fraud / total * 100) if total else 0, 2), average_amount=round(float(average), 2), risk_distribution=risks, recent=recent)

@dashboard_bp.get("/transactions")
def transaction_list():
    query = Transaction.query
    if request.args.get("prediction") in {"Legitimate", "Fraudulent"}: query = query.filter_by(prediction=request.args["prediction"])
    if request.args.get("risk_level") in {"LOW", "MEDIUM", "HIGH"}: query = query.filter_by(risk_level=request.args["risk_level"])
    term = request.args.get("q", "").strip()
    if term:
        query = query.filter(Transaction.id.cast(__import__('sqlalchemy').String).like(f"%{term}%"))
    page = max(request.args.get("page", 1, type=int), 1)
    result = query.order_by(Transaction.created_at.desc()).paginate(page=page, per_page=20, error_out=False)
    return jsonify(items=[row.to_dict() for row in result.items], page=page, pages=result.pages, total=result.total)

@dashboard_bp.get("/model-performance")
def model_performance():
    metrics = load_metrics()
    if not metrics: return jsonify(error="No model metrics available. Train the model first."), 404
    return jsonify(metrics)
