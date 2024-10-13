from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('model/ridge.pkl','rb'))
scaler_model=pickle.load(open('model/scaler.pkl','rb'))

@app.route("/")


def index():
    return render_template("index.html")

@app.route("/predictdata",methods=['GET','POST'])

def predict_data():
    if request.method=="POST":
        Present_Price=float(request.form.get('Present_Price'))
        Kms_Driven=float(request.form.get('Kms_Driven'))
        Fuel_Type=float(request.form.get('Fuel_Type'))
        Seller_Type=float(request.form.get('Seller_Type'))
        Transmission=float(request.form.get('Transmission'))
        Owner=float(request.form.get('Owner'))

        new_data_scale=scaler_model.transform([[Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner]])

        result=ridge_model.predict(new_data_scale)

        return render_template("home.html",results=result[0])
    else:
        return render_template("home.html")


if __name__=="__main__":
    application.run(host="0.0.0.0")