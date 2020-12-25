
import numpy as np
import pandas as pd
from flask import Flask, request, render_template 
import pickle
#import os
import xgboost as xgb

app = Flask(__name__, static_folder='./static', template_folder="./templates")

#load saved model
def load_model():
    print("Loading saved model...[Start]")
    bst = xgb.Booster()  # init model

    #pickle.load(open('artifacts/pkl_xgb_model.pkl', 'rb'))
    return bst.load_model('artifacts/xgb_model.json')

#home page
@app.route('/')
def home():
    return render_template('index.html')

#predict the result and return
@app.route('/predict', methods=['POST'])
def predict():

    labels = ['Rejected', 'Approved']
    cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
           'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    cat_colst = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    features=[x for x in request.form.values()]

    data = pd.DataFrame([np.array(features)], columns = cols)
    features = pd.get_dummies(data, columns=cat_colst)

    values = [np.array(features)]

    model = load_model()
    print("Loaded saved model...[End]")
    prediction = model.predict(values)

    result = labels[prediction[0]]

    return render_template('index.html', output='Loan {}'.format(result))
    
if __name__ == "__main__":
    #port = int(os.environ.get('PORT', 5000))
    #app.run(port=port,debug=True,use_reloader=False)
    app.run(debug=False)