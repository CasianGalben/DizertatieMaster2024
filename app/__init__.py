from flask import Flask, request
from .extensions import db
from .routes import main
from flask_migrate import Migrate
from .naive_bayes_api import naive_bayes_blueprint
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-very-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../data/carti.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    UPLOAD_FOLDER = 'app/static/uploads'
    STATIC_FOLDER = 'static'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['STATIC_FOLDER'] = STATIC_FOLDER

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

    db.init_app(app)
    migrate = Migrate(app, db)

    app.register_blueprint(main)
    app.register_blueprint(naive_bayes_blueprint, url_prefix='/naive_bayes')

    return app
