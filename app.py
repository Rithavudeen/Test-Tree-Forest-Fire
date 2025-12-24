from flask import Flask, render_template, request
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import pickle


app = Flask(__name__)

ridgecv_model=pickle.load(open('models/ridgecv.pkl', 'rb'))
Standard_Scaler=pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
            Temperature = float(request.form.get('Temperature'))
            RH= float(request.form.get('RH'))
            Ws= float(request.form.get('Ws'))
            Rain= float(request.form.get('Rain'))
            FFMC= float(request.form.get('FFMC'))
            DMC= float(request.form.get('DMC'))
            ISI= float(request.form.get('ISI'))
            Classes= float(request.form.get('Classes'))
            Region= float(request.form.get('Region'))
            new_scaled_data=Standard_Scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
            prediction=ridgecv_model.predict(new_scaled_data)
            output=round(prediction[0],2)
      
    return render_template('index.html', prediction_text=f"Predicted Fire Weather Index: {output}")

if __name__ == "__main__":
    app.run(debug=True)
