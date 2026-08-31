import logging
from pathlib import Path
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from config import Config
from models.database_models import db


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    db.init_app(app)
    with app.app_context():
        db.create_all()

    from routes.main_routes import main_bp
    from routes.prediction_routes import prediction_bp
    from routes.dashboard_routes import dashboard_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(dashboard_bp)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="Upload is too large."), 413

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
    return app


if __name__ == "__main__":
    create_app().run(debug=True, host="127.0.0.1", port=5000)
