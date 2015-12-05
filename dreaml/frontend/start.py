from flask import Flask
from flask_bootstrap import Bootstrap

from frontend import construct_frontend,nav

def create_app(df=None):
    app = Flask(__name__)
    Bootstrap(app)

    frontend = construct_frontend(df)

    app.register_blueprint(frontend)

    nav.init_app(app)

    return app