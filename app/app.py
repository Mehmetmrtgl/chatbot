from flask import Flask
from flask_cors import CORS
from app.db.models import db
from app.routes.chat_routes import chat_bp
from app.routes.feedback_routes import feedback_bp
from app.routes.autocomplete_routes import autocomplete_bp
from app.routes.admin_routes import admin_bp
from app.routes.chat_history_routes import chat_history_bp
from app.routes.suggested_answers_routes import suggestion_bp
def create_app():
    app = Flask(__name__)
    CORS(app)  # 👈 Bu satırla tüm endpoint'lere CORS açılır

    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "postgresql://postgres:123456@localhost:5432/hu_chatbot2"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    app.register_blueprint(chat_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(autocomplete_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(chat_history_bp)


    app.register_blueprint(suggestion_bp)

    return app



if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5001)
