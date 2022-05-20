from flask import Flask, request, render_template
from flaskext.markdown import Markdown
import flask_config
import comment_filter
import spacy
from spacy import displacy

spacy.require_gpu()

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

app = Flask(__name__, template_folder = 'template')
Markdown(app)

# Limit upload file size to 1 megabyte
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024


@app.errorhandler(404)
def page_not_found(e):
    return '<h1>404</h1><p>The resource could not be found.</p>', 404


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods = ['POST'])
def predict():

    input_text = request.form.get('textbox')
    
    # return render_template('home.html', display_right = True, input = input_text, render_result = [False, False, None])

    # displaCy visualizer for NER
    ner_nlp = spacy.load(flask_config.ner_model)
    ner_doc = ner_nlp(input_text)

    ner_options = {'colors': {'ORG': '#ffcc33', 'LOC': '#fb836f', 'PER': '#43c6fc', 'MISC': '#a6e22d'}}
    ner_tags = displacy.render(ner_doc, style='ent', options=ner_options)
    ner_tags.replace('\n\n', '\n')
    render_input = ner_tags

    model = request.form.get('models')

    nlp = spacy.load(getattr(flask_config, model + '_model'))

    offensive_threshold = getattr(flask_config, model + '_offensive_threshold')
    targeted_threshold = getattr(flask_config, model + '_targeted_threshold')

    render_thresholds = [offensive_threshold, targeted_threshold]

    is_edmw = False

    if model == 'reddit' or model == 'hwz':
        is_edmw = True

    filtered_text = comment_filter.c_filter(
        uncased=False,
        edmw=is_edmw,
        input_list=[input_text])

    doc = nlp(filtered_text[0])

    result = doc.cats
    is_off = False
    is_tin = False
    target = None

    if result['offensive'] > offensive_threshold:
        is_off = True

        if result['targeted'] > targeted_threshold:
            is_tin = True

            result.pop('offensive')
            result.pop('targeted')
            prediction = max(result, key=result.get)

            if prediction == 'individual':
                target = 'IND'
            elif prediction == 'group':
                target = 'GRP'
            else:
                target = 'OTH'

    render_result = [is_off, is_tin, target]

    return render_template('home.html', display_right = True, input = render_input, result_array = render_result, threshold_array = render_thresholds)


if __name__ == '__main__':
    app.run(debug=True)