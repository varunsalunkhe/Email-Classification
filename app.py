#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import joblib
import traceback
from flask import Flask, request, jsonify, render_template, url_for

app=Flask(__name__)

model=joblib.load("model.pkl")
print("model loaded")

word_dict = joblib.load("word_dict.pkl")
print("word_dict loaded")

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/predict", methods=["GET","POST"])

def predict():
    if model:
        try:
            # json_ = request.json

            json_ = request.form.get("text")

            print(json_)
            
            mail = []
            for i in word_dict:
                mail.append(json_.split(" ").count(i[0]))
                
            sample = np.array(mail).reshape(1,3000)

           

            prediction= model.predict(sample)

            if prediction == 0:


                # return jsonify({"Prediction ": "spam"})
                render_template('index.html', Prediction= "spam")

            else:
                render_template('index.html', Prediction= "not spam")
                # return jsonify({"Prediction ": "not spam"})



        except:
            return jsonify({"trace ": traceback.format_exc()})
    else:
        print("first train the model")
        return ("no model is here to use")

if __name__ == "__main__":
    app.run(debug= True)







