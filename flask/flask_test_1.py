#Import main library
import numpy as np

#Import Flask modules
from flask import Flask, request, render_template

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'template')
app.config["DEBUG"] = True

# Limit upload file size to 2 megabytes
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

#Open our model 
#create our "home" route using the "index.html" page
@app.route('/')
def home():
    return render_template('index.html')

#Set a post method to yield predictions on page
@app.route('/', methods = ['POST'])
def predict():
    
    nlp_string = request.form.values()

    if output < 0:
        return render_template('index.html', prediction_text = "Predicted Price is negative, values entered not reasonable")
    elif output >= 0:
        return render_template('index.html', prediction_text = 'Predicted Price of the house is: ${}'.format(output))   

#Run app
if __name__ == "__main__":
    app.run(debug=True)