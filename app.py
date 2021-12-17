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
    confidence = model.predict_proba(vect)
    
    

    if prediction[0]:
        output = 'Positive'
        cofd = round((confidence.flat[1]*100),2)
    else:
        output = 'Negative'
        cofd = round((confidence.flat[0]*100),2)

    return render_template('Index.html', prediction_text='Sentiment Polarity: {}'.format(output), confidence_text='Confidence: {}'.format(cofd))


if __name__ == "__main__":
    app.run(debug=True)
    app.config["DEBUG"]=True
