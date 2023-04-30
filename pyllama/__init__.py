import os

from flask import Flask

from . import constants, download, prompt
import secrets


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    app.secret_key = secrets.token_hex()
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    if not os.path.isdir(constants.MODEL_PATH):
        raise FileNotFoundError(f"Models directory: {constants.MODEL_PATH}, does not exist!")

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
        os.makedirs(constants.MODEL_PATH)
    except OSError:
        pass

    app.register_blueprint(download.bp)
    app.register_blueprint(prompt.bp)
    return app
