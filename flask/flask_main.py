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
    
    return render_template('home.html', display_right = True, input = input_text, render_result = [False, False, None])

    # model = request.form.get("models")

    # nlp = spacy.load(getattr(flask_config, model + '_model'))

    # is_edmw = False

    # if model == 'reddit' or model == 'hwz':
    #     is_edmw = True

    # filtered_text = comment_filter.c_filter(
    #     uncased=False,
    #     edmw=is_edmw,
    #     input_list=[input_text])

    # doc = nlp(filtered_text[0])
    # result = doc.cats

    # is_off = False
    # is_tin = False
    # target = None

    # if result["offensive"] > flask_config.offensive_threshold:
    #     is_off = 1

    #     if result["targeted"] > flask_config.targeted_threshold:
    #         is_tin = True

    #         result.pop('offensive')
    #         result.pop('targeted')
    #         prediction = max(result, key=result.get)

    #         if prediction == 'individual':
    #             target = 'IND'
    #         elif prediction == 'group':
    #             target = 'GRP'
    #         else:
    #             target = 'OTH'

    # result_array = [is_off, is_tin, target]

    # return render_template('home.html', display_right = True, input = input_text, render_result = result_array)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
    
@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == "__main__":
    app.run(debug=True)