import glob
import re

from flask import Blueprint, current_app, render_template, request, session
from llama_cpp import Llama
from yaml import safe_load

from pyama import constants, download

bp = Blueprint('prompt', __name__)

responses = []
MODEL = None

CONFIG_TYPE_CASTS = {
    'max_tokens': int,
    'top_k': int,
    'top_p': float,
    'temperature': float,
    'repeat_penalty': float,
    'logprobs': int,
    'stop_strings': str,
    'debug': bool
}


def dummy_model(prompt, **kwargs):
    return f"I should respond to {prompt} with {kwargs}"


def initialize_model(model_path, **kwargs):
    global MODEL
    current_app.logger.info(f"Attempting to load model from {model_path}")
    try:
        MODEL = Llama(model_path, n_ctx=2000, **kwargs)
    except Exception as e:
        current_app.logger.error(f"{e}")
        MODEL = dummy_model


def get_prompt_patterns(prompt_path=constants.PROMPT_PATH):
    patterns = {}
    pattern_files = glob.glob(prompt_path+"/*.yaml")
    current_app.logger.info(f"Discovered prompt patterns: {pattern_files}")

    # Safely load all yaml files in the directory
    for pattern_file in pattern_files:
        with open(pattern_file, 'r') as pattern_file:
            pattern = safe_load(pattern_file.read())
            patterns[pattern['name']] = pattern.copy()
    return patterns


def render_prompt(prompt_pattern, user_input):
    """Replace substitue tokens from the prompt pattern with user input
    and model output.
    """
    prompt_pattern = re.sub('<PROMPT>', user_input, prompt_pattern)
    prompt_pattern = re.sub('<RESPONSE>', '', prompt_pattern)
    return prompt_pattern


def get_settings(prompt_patterns):

    model_settings = request.form.copy()
    runtime_settings = {
        'selected_prompt': model_settings.pop('selected_prompt'),
        'selected_model': model_settings.pop('selected_model')
    }
    # Remove some keys from persistent settings
    model_settings.pop('prompt')
    selected_prompt = runtime_settings['selected_prompt']

    for key, value in request.form.items():
        if key in model_settings:
            if len(model_settings[key]) == 0:
                model_settings.pop(key)
            elif key == 'logprobs' and 'logits_all' not in model_settings:
                model_settings.pop(key)
            else:
                model_settings[key] = CONFIG_TYPE_CASTS[key](value)

    # Default to prompt defined stop strings if necessary
    if 'stop_strings' not in model_settings:
        stop_strings = prompt_patterns[selected_prompt].get('stop_strings', [])
        current_app.logger.info(f"Defaulting to base prompt stop strings {stop_strings}")
        model_settings['stop_strings'] = ','.join(stop_strings)
    return model_settings, runtime_settings


def get_response(prompt='', debug=False, max_tokens=256, stop_strings=None,
                 prompt_pattern='', **kwargs):
    global MODEL
    stop_strings = stop_strings.split(',')
    rendered_prompt = render_prompt(prompt_pattern, prompt)

    current_app.logger.info(
        f"Prompt rendered as: {rendered_prompt}")

    model_out = MODEL(
        rendered_prompt, max_tokens=max_tokens,
        stop=stop_strings, echo=True, **kwargs)
    if debug:
        return rendered_prompt, model_out
    model_out = model_out['choices'][0]['text']
    model_out = model_out[len(rendered_prompt):]
    return prompt, model_out


@bp.route('/', methods=['GET', 'POST'])
@bp.route('/prompts', methods=['GET', 'POST'])
def prompts():
    available_models = download.get_models_list()
    available_prompts = get_prompt_patterns()
    model_path = request.form.get('selected_model', "no_model")
    selected_prompt = request.form.get('selected_prompt', "")

    # Logging model settings
    current_app.logger.info(f"Prompts {available_prompts}")
    current_app.logger.info(f"Selected model path: {model_path}")

    if 'model_settings' in session.keys():
        current_app.logger.info(f"Session settings {session['model_settings']}")
    if 'selected_prompt' in session.keys():
        current_app.logger.info(f"Selected prompt: {session['selected_prompt']}")

    if request.method == 'POST':
        if model_path == "no_model":
            current_app.logger.warning("No model selected")
            responses.append("No model selected!")
        else:
            initialize_model(model_path)
            model_settings, runtime_settings = get_settings(available_prompts)

            # Preserving model settings and prompt between runs as a personalized cookie
            session['model_settings'] = model_settings.copy()
            session['runtime_settings'] = runtime_settings.copy()
            session['selected_prompt'] = selected_prompt
            session.modified = True

            selected_prompt = available_prompts[selected_prompt]

            current_app.logger.info(f"Current model settings: {model_settings}")
            responses.append(
                get_response(prompt=request.form['prompt'],
                             prompt_pattern=selected_prompt['pattern'], **model_settings))

    return render_template(
        'prompts.html', responses=reversed(responses), available_models=available_models,
        model_settings=session.get('model_settings', {}), available_prompts=available_prompts,
        runtime_settings=session.get('runtime_settings', {}))
