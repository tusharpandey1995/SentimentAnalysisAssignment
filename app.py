import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))
tv = pickle.load(open('tv_transform.pkl', 'rb'))

@app.route('/')
def home():

    return render_template('Index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    message = request.form['inputText']
    data = [message]
    vect = tv.transform(data).toarray()
    prediction = model.predict(vect)

    output = 'Positive' if prediction[0] else 'Negative'

    return render_template('Index.html', prediction_text='Sentiment Polarity: '.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    app.config["DEBUG"]=True
