import pandas as pd
import numpy as pd
import joblib
import traceback
from flask import Flask, request, jsonify

app=Flask(__name__)

model=joblib.load("model.pkl")
print("model loaded")

word_dict = joblib.load("word_dict.pkl")
print("word_dict loaded")


@app.route("/mail", methods=["GET","POST"])

def predict():
	if model:
		try:
			json_ = request.json

            mail= []

            for i in word_dict:
        		mail.append(json_.split(" ").count(i[0]))

            sample = np.array(mail)

			prediction= list(model.predict(sample))

			return jsonify({"Prediction ": str(prediction)})	
				

		except:
			return jsonify({"trace ": traceback.format_exc()})
	else:
		print("first train the model")
		return ("no model is here to use")

if __name__ == "__main__":
	app.run(debug= True)