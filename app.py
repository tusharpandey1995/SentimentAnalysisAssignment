import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sys
# Libraries for feature engineering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def home():

    return render_template('Index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # str_features = [str(x) for x in request.fm.values()]
    # print(str_features)
    print('+++++++App Running from predict+++++++')
    tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1,2),sublinear_tf=True)
    str_features = pd.DataFrame([np.array(request.form['inputText'])])
    print('+++++++str_features+++++++',np.array(request.form['inputText']))
    final_features = tv.fit_transform(str_features)
    # final_features = 
    print(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('Index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    app.config["DEBUG"]=True

