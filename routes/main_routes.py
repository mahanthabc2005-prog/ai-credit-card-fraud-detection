from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)

@main_bp.get("/")
def home(): return render_template("index.html")

@main_bp.get("/detect")
def detect(): return render_template("detect.html")

@main_bp.get("/batch-detection")
def batch(): return render_template("batch.html")

@main_bp.get("/dashboard")
def dashboard(): return render_template("dashboard.html")

@main_bp.get("/transactions")
def transactions(): return render_template("transactions.html")

@main_bp.get("/model-performance")
def performance(): return render_template("model_performance.html")

@main_bp.get("/about")
def about(): return render_template("about.html")
