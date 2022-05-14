from flask import Flask, request, render_template
import main_config
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

    if model == olid:
        # nlp = spacy.load(olid_model)
        return render_template('index.html', prediction_text = str(1))

    elif model == reddit:
        # nlp = spacy.load(reddit_model)
        return render_template('index.html', prediction_text = str(2))

    elif model == hwz:
        # nlp = spacy.load(hwz_model)
        return render_template('index.html', prediction_text = str(3))



    # return render_template('index.html', prediction_text = str(input_text))


if __name__ == "__main__":
    app.run(debug=True)