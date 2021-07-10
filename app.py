
"""
Created on Sat Jun 12 15:59:32 2021
@author: khurram
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

lr = joblib.load("data_predicted.pkl")

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def main():
    if request.method == 'POST':
        # DATE
        date = request.form['date']
        day = float(pd.to_datetime(date, format="%Y-%m-%dT").day)
        month = float(pd.to_datetime(date, format="%Y-%m-%dT").month)
		# PM10_24hr_avg
        PM10_24hr_avg = float(request.form['PM10_24hr_avg'])
		# SO2_24hr_avg
        SO2_24hr_avg = float(request.form['SO2_24hr_avg'])
		# NO2_1hr_avg
        NO2_1hr_avg = float(request.form['NO2_1hr_avg'])
		# CO_8hr_max
        CO_8hr_max = float(request.form['CO_8hr_max'])
		# O3_8hr_max
        O3_8hr_max = float(request.form['O3_8hr_max'])
		# Station
        Station= float(request.form['Station'])
        
        my_prediction = lr.predict([[Station,PM10_24hr_avg,SO2_24hr_avg,NO2_1hr_avg,CO_8hr_max, O3_8hr_max,month,day]])
    
        my_prediction1= int(my_prediction[:,:1])
        
        my_prediction2 = int(my_prediction[:,1:2])
        
        my_prediction3 = my_prediction[:,2:3]
        
        
        
    return render_template('index.html', my_prediction1=np.round([my_prediction1]), my_prediction2=np.round([my_prediction2]), my_prediction3=[my_prediction3])

if __name__ == "__main__":
    app.run(debug = True)