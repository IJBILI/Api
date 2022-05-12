# Dependencies
from os import system
from flask import Flask, request, jsonify

import pickle
import traceback
import pandas as pd
import numpy as np
#import prediction_api
#from prediction_api import *

# Your API definition
app = Flask(__name__)
randomForest = pickle.load(open('./classifier_rf_model.sav', 'rb'))
sample_size = 10000
data_set = pd.read_csv('./data_test.csv',nrows=sample_size)

@app.route('/predictByClientId', methods=['POST'])
 
def predictByClientId():
   
    if randomForest:
        try:
            json_ = request.json
            print(json_)
            
            client=data_set[data_set['SK_ID_CURR']==int(json_['SK_ID_CURR'])].drop(['SK_ID_CURR','TARGET'],axis=1)
            print(client)
            
            y_pred = randomForest.predict(client)
            y_proba = randomForest.predict_proba(client)
            
            return jsonify({'prediction': str(y_pred[0]),'prediction_proba':y_proba[0][0]})


        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')

