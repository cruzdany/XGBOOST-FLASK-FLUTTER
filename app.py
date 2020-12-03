from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback
from flask_restful import reqparse

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	lr = joblib.load("model.pkl")
	if lr:
		try:
			json = request.get_json()
			#print(json)	 
			model_columns = joblib.load("model_cols.pkl")
			#print(model_columns)
			temp=list(json[0].values())
			vals=np.array(temp)
			prediction = lr.predict(temp)
			print("here:",prediction)        
			return jsonify({'prediction': str(prediction[0])})

		except:        
			return jsonify({'trace': traceback.format_exc()})
	else:
		return ('No model here to use')
    
if __name__ == '__main__':
    app.run(debug=True)