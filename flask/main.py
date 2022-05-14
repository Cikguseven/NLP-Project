from flask import Flask, request, render_template
import main_config
import comment_filter
import spacy

spacy.require_gpu()

app = Flask(__name__, template_folder = 'template')

# Limit upload file size to 2 megabytes
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods = ['GET', 'POST'])
def predict():
    
    model = request.form.get("models")

    input_text = request.form.get("sentence")

    nlp = spacy.load(main_config.globals()[model + '_model'])

    if input_text:
        filtered_text = comment_filter.c_filter(
            shuffle=False,
            remove_username=False,
            remove_commas=False,
            length_min=0,
            length_max=999999999,
            uncased=False,
            unique=False,
            edmw=False
            input_list=[input_text])
        doc = nlp(input_text)



    return render_template('index.html', prediction_text = str(3))


if __name__ == "__main__":
    app.run(debug=True)