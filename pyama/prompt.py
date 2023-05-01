import logging
import os

from flask import (Blueprint, Flask, current_app, render_template, request,
                   session)
from llama_cpp import Llama

from pyama import constants, download

bp = Blueprint('prompt', __name__)

responses = []
MODEL = None

CONFIG_CASTS = {
    'max_tokens': int,
    'top_k': int,
    'top_p': float,
    'temperature': float,
    'repeat_penalty': float,
    'logprobs': int,
    'stop_strings': str
}

def dummy_model(prompt, **kwargs):
    return f"I should respond to {prompt} with {kwargs}"

def initialize_model(model_path, **kwargs):
    global MODEL
    current_app.logger.info(f"Attempting to load model from {model_path}")
    try:
        MODEL  = Llama(model_path, **kwargs)
    except Exception as e:
        current_app.logger.error(f"{e}")
        MODEL = dummy_model

def get_settings():

    model_settings = request.form.copy()
    model_settings.pop('prompt')
    model_settings.pop('selected_model')
    for key, value in request.form.items():
        if key in model_settings:
            if len(model_settings[key]) == 0:
                model_settings.pop(key)
            elif key == 'logprobs' and 'logits_all' not in model_settings:
                model_settings.pop(key)
            else:
                model_settings[key] = CONFIG_CASTS[key](value)

    return model_settings

def get_response(prompt='', debug=False, max_tokens=256, stop_strings=None, **kwargs):
    global MODEL
    model_out = MODEL(prompt, max_tokens=max_tokens, stop=stop_strings, echo=True, **kwargs)
    if debug:
        return model_out
    return model_out['choices'][0]['text']

@bp.route('/', methods=['GET', 'POST'])
@bp.route('/prompts', methods=['GET', 'POST'])
def prompts():
    available_models = download.get_models_list()
    model_path = request.form.get('selected_model', "no_model")
    current_app.logger.info(f"Selected model path: {model_path}")
    if 'model_settings' in session.keys():
        current_app.logger.info(f"Session settings {session['model_settings']}")
    if request.method == 'POST':
        if model_path == "no_model":
            current_app.logger.warning("No model selected")
            responses.append("No model selected!")
        else:
            initialize_model(model_path)
            model_settings = get_settings()
            session['model_settings'] = model_settings.copy()
            session.modified = True
            current_app.logger.info(f"Current model settings: {model_settings}")
            responses.append(
                get_response(prompt=request.form['prompt'], debug=request.form.get('debug', False), **model_settings))

    return render_template(
        'prompts.html', responses=reversed(responses), available_models=available_models,
        model_settings=session.get('model_settings', {}))
