from flask import Flask, render_template, request
#import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('linear_reg_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def main():
    if request.method == 'POST':
        T, TM, Tm, SLP, H, VV, V, VM = float(request.form['T']), float(request.form['TM']), float(request.form['Tm']), float(request.form['SLP']), float(request.form['H']), float(request.form['VV']), float(request.form['V']), float(request.form['VM'])
        lr_pm = model.predict([[T, TM, Tm, SLP, H, VV, V, VM]])




    return render_template("index.html", lr_pm = np.round(lr_pm,3))

if __name__ == "__main__":
    app.run(debug = True)
