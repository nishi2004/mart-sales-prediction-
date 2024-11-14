from flask import Flask,request, url_for, redirect, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    Item_Weight = float(data['Item_Weight'])
    Item_Visibility = float(data['Item_Visibility'])
    Item_MRP = float(data['Item_MRP'])
    Outlet_Establishment_Year = float(data['Outlet_Establishment_Year'])
    
    prediction = model.predict([[Item_Weight, Item_Visibility, Item_MRP, Outlet_Establishment_Year]])
    
    return jsonify({'predictions': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
