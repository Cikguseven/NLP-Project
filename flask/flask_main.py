from flask import Flask, request, render_template
import flask_config
import comment_filter
import spacy

spacy.require_gpu()

app = Flask(__name__, template_folder = 'template')

# Limit upload file size to 1 megabyte
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods = ['GET', 'POST'])
def predict():

    input_text = request.form.get("textbox")
    
    model = request.form.get("models")

    nlp = spacy.load(getattr(flask_config, model + '_model'))

    is_edmw = False

    if model == 'reddit' or model == 'hwz':
        is_edmw = True

    filtered_text = comment_filter.c_filter(
        uncased=False,
        edmw=is_edmw,
        input_list=[input_text])

    doc = nlp(filtered_text[0])

    if doc.cats['offensive'] > 0.6:
        result = 'offensive'
    else:
        result = 'not offensive'

    return render_template('home.html', display_right = True, input = input_text, result = 'Sentence is ' + result)


if __name__ == "__main__":
    app.run(debug=True)