import glob
import os
import tempfile

import pycurl
from flask import Blueprint
from flask import blueprints as bp
from flask import current_app, render_template, request
from git import Repo

from pyama.constants import MODEL_PATH

bp = Blueprint('models', __name__)

def get_models_list(path=MODEL_PATH):
    model_paths = [
        m
        for m
        in glob.glob(os.path.join(path, '**/*.bin',), recursive=True)
        if m.endswith('.bin')]
    return model_paths

def clone_repo(url):

    current_app.logger.info(f"Cloning repo from {url}")
    with tempfile.TemporaryDirectory() as tmpdir:
        Repo.clone_from(url, tmpdir)


def curl_model(url):
    c = pycurl.Curl()
    with open(os.path.join(MODEL_PATH, url.split('/')[-1]), mode='wb') as model_file:
        c.setopt(c.WRITEDATA, model_file)
        c.setopt(c.URL, url)
        c.setopt(c.FOLLOWLOCATION, True)
        c.perform()
        c.close()

def download_model(url):
    current_app.logger.info(f"Downloading model from {url}")

    if url.endswith('.git'):
        clone_repo(url)
    else:
        curl_model(url)

@bp.route('/models', methods=['POST', 'GET'])
def models(model_path=MODEL_PATH):
    errors = []

    if request.method == 'POST':
        try:
            download_model(request.form['model_url'])
        except Exception as e:
            errors.append(e)
            current_app.logger.error(f"{e}")
    return render_template('models.html', models=get_models_list(model_path), errors=errors)
