from flask import Flask, jsonify, request
from flask_cors import CORS
from flaskext.markdown import Markdown
from spacy import displacy
import comment_filter
import flask_config
import re
import spacy

spacy.require_gpu()

app = Flask(__name__)
CORS(app)
Markdown(app)


def pipeline(input_sentence: str, model: str):

    # displaCy visualizer for NER
    ner_nlp = spacy.load(flask_config.ner_model)

    stopwords = ner_nlp.Defaults.stop_words
    stopwords.add('to be')
    stopwords.add('u')

    ner_doc = ner_nlp(input_sentence)
    ner_doc.ents = tuple(x for x in ner_doc.ents if x.text.lower() not in stopwords)
    
    output_ner_tags = displacy.render(ner_doc, style='ent', options={'colors': {'ORG': '#ffd24d', 'LOC': '#fc9583', 'PER': '#4fc8fc', 'MISC': '#a9e335'}})
    output_ner_tags.replace('\n\n', '\n')

    nlp = spacy.load(getattr(flask_config, model + '_model'))

    offensive_threshold = getattr(flask_config, model + '_offensive_threshold')
    targeted_threshold = getattr(flask_config, model + '_targeted_threshold')

    filtered_text = comment_filter.c_filter(
        edmw=True if model == 'reddit' or model == 'hwz' else False,
        input_list=[input_sentence])

    doc = nlp(filtered_text[0])

    result = doc.cats
    is_off = 'No'
    is_tin = 'Not Applicable'
    target = 'Not Applicable'

    if result['offensive'] > offensive_threshold:
        is_off = 'Yes'

        if result['targeted'] > targeted_threshold:
            is_tin = 'Yes'

            result.pop('offensive')
            result.pop('targeted')
            prediction = max(result, key=result.get)

            if prediction == 'individual':
                target = 'Individual'
            elif prediction == 'group':
                target = 'Group'
            else:
                target = 'Other'

        else:
            is_tin = 'No'

    return output_ner_tags, is_off, is_tin, target

@app.route('/predict', methods=['POST'])
def predict():
    form_text = request.form['textbox']
    form_model = request.form['models']

    if re.search('[a-zA-Z]', form_text.strip()):
        ner_tagged_input, offensive_result, targeted_result, target_result = pipeline(form_text, form_model)

        return jsonify({"tagged_input": ner_tagged_input, "offensive": offensive_result, "targeted":targeted_result, "target":target_result})

    return jsonify({"error": True})

if __name__ == '__main__':
    app.run(debug=True)