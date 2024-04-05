import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow
from recom_f_promt import recommend_categ_for_prompt
from recom_f_ann import predict_ann

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #new_data = [str(x) for x in request.form.values()]
    inp = str(request.form['news'])
    pred = recommend_categ_for_prompt(inp)
    return render_template('index.html', prediction_text='Business Recommendation should be {}'.format(pred))

@app.route('/extra_predict',methods=['POST'])
def extra_predict():
    '''
    For rendering results on HTML GUI
    '''
    prompt = str(request.form['news'])
    location = str(request.form['location'])
    size = str(request.form['size'])
    pred = predict_ann(prompt,location,size)
    return render_template('index.html', prediction_text='Business Recommendation should be {}'.format(pred))

if __name__ == "__main__":
    app.run(debug=True)