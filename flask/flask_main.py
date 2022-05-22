from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory
from flaskext.markdown import Markdown
from spacy import displacy
from werkzeug.utils import secure_filename
import comment_filter
import flask_config
import spacy
import os
from flask_wtf import FlaskForm
from wtforms import FileField, TextAreaField, RadioField
from wtforms.validators import InputRequired, Length, Regexp, Optional
import re

spacy.require_gpu()


class UserControlForm(FlaskForm):
    model = RadioField('Select model:',
                       validators=[InputRequired()],
                       choices=[('olid', 'Trained on OLID'), ('hwz', 'Trained on HWZ EDMW'), ('reddit', 'Trained on r/Singapore')],
                       default='olid')
    user_input = TextAreaField('Input sentence:', validators=[Optional()], default="Lee Kuan Yew (16 September 1923 – 23 March 2015), born Harry Lee Kuan Yew, often referred to by his initials LKY and in his earlier years as Harry Lee, was a Singaporean statesman and lawyer who served as the first prime minister of Singapore between 1959 and 1990. He is widely recognised as the nation's founding father. Lee was born in Singapore during British colonial rule, which was then part of the Straits Settlements. He gained an educational scholarship to Raffles College, and during the Japanese occupation, he worked in private enterprises and as an administration service officer for the propaganda office. After the war, Lee attended the London School of Economics, but transferred to Fitzwilliam College, Cambridge, graduating with starred-first-class honours in law in 1947. He became a barrister of the Middle Temple in 1950 and campaigned for the United Kingdom to relinquish its colonial rule upon returning to Singapore.")
    user_file = FileField('Upload file:', validators=[Optional(), Regexp(r'^[^/\\]\.(json|csv)$', flags=re.IGNORECASE)])


app = Flask(__name__, template_folder = 'template')
Markdown(app)

# Limit upload file size to 1 megabyte
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


@app.errorhandler(404)
def page_not_found(e):
    return '<h1>404</h1><p>The resource could not be found.</p>', 404


def pipeline(input_sentence: str, model: str):
    # displaCy visualizer for NER
    ner_nlp = spacy.load(flask_config.ner_model)
    ner_doc = ner_nlp(input_sentence)
    output_ner_tags = displacy.render(ner_doc, style='ent', options={'colors': {'ORG': '#ffd24d', 'LOC': '#fc9583', 'PER': '#4fc8fc', 'MISC': '#a9e335'}})
    output_ner_tags.replace('\n\n', '\n')

    nlp = spacy.load(getattr(flask_config, model + '_model'))

    offensive_threshold = getattr(flask_config, model + '_offensive_threshold')
    targeted_threshold = getattr(flask_config, model + '_targeted_threshold')

    is_edmw = False

    if model == 'reddit' or model == 'hwz':
        is_edmw = True

    filtered_text = comment_filter.c_filter(
        uncased=False,
        edmw=is_edmw,
        input_list=[input_sentence])

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

    output_result = [is_off, is_tin, target]

    return output_ner_tags, output_result


@app.route('/', methods = ['GET', 'POST'])
def predict():

    form = UserControlForm()

    if form.validate_on_submit():
        received_model = form.model.data
        received_text = form.user_input.data
        received_file = form.user_file.data
        
        ner_tagged_input, render_result = pipeline(received_text, received_model)

        return render_template('home.html', display_right = True, input = ner_tagged_input, result_array = render_result, form=form)

    return render_template('home.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)