from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_time = db.Column(db.Float, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    fraud_probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(12), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id, "transaction_time": self.transaction_time, "amount": self.amount,
            "prediction": self.prediction, "fraud_probability": round(self.fraud_probability, 4),
            "risk_level": self.risk_level, "created_at": self.created_at.isoformat(),
        }
