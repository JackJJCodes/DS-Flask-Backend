from distutils.log import debug
import mimetypes
from flask import Flask, request, send_from_directory, url_for, redirect, render_template, jsonify
import pandas as pd
import os
import pickle
from flask_cors import CORS, cross_origin

app = Flask(__name__)
# CORS(app)

model = pickle.load(open("example_weights_log.pkl", "rb"))

@app.route('/')
@cross_origin(supports_credentials=True)
def home():
    return render_template("index.html")

@app.route('/favicon.ico')
@cross_origin(supports_credentials=True)
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def predict():
    # to take Json input for API call
    json_data = request.get_json()
    print(json_data)
    input_1 = int(json_data['1'])
    input_2 = int(json_data['2'])
    input_3 = int(json_data['3'])
    input_4 = int(json_data['4'])
    input_5 = int(json_data['5'])
    input_6 = int(json_data['6'])
    input_7 = int(json_data['7'])
    input_8 = int(json_data['8'])

    # input_1 = request.form['1']
    # input_2 = request.form['2']
    # input_3 = request.form['3']
    # input_4 = request.form['4']
    # input_5 = request.form['5']
    # input_6 = request.form['6']
    # input_7 = request.form['7']
    # input_8 = request.form['8']

    setup_df = pd.DataFrame([pd.Series([input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8])])
    diabetes_prediction = model.predict_proba(setup_df)
    output = '{0:.{1}f}'.format(diabetes_prediction[0][1], 2)
    output = str(float(output) * 100) + '%'
    if output > str(0.5):
        # return render_template('result.html', pred = "You have a chance of getting diabetes. Your score is {output}")
        return jsonify({"pred": "You have a chance of getting diabetes. Your score is "+output })
    else:
        # return render_template('result.html', pred = "You don't seem to have diabetes. Your score is {output}")
        return jsonify({"pred": "You don't seem to have diabetes. Your score is "+output })
    
if __name__ == '__main__':
    app.run(debug=True)