import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sys
# Libraries for feature engineering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

    output = round(prediction[0], 2)

    return render_template('Index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    app.config["DEBUG"]=True

