from flask import Flask
from flask import jsonify
from flask import request

import json

#MODEL RELATED
from defeasible_clf_model import *

app = Flask(__name__)

print("Loading Defeasible CLF ATOMIC")
clf_model = DefeasibleClfModel(data_type="atomic")

#Demo
# @app.route('/')
# def hello():
#     return 'Hello World!'

#Receives Parameter from gunicorn
# def load_app(gpu_num):
#     cfg = load_app_config(cfg_file)
#     return my_app(cfg)


#Only accept POST Request
@app.route('/predict_clf', methods=['POST'])
def predict_clf():
    if request.method == 'POST':
        # Get received JSON
        recv_json=request.json  
        print(recv_json)

        # premise=recv_json["premise_input"]
        # hypothesis=recv_json["hyp_input"]
        # update=recv_json["update_input"]

    sample_premise="[MALE] has a very important math test next week."
    sample_hypothesis="[MALE] get a good grade on the test."
    sample_update="The math test is very difficult"

    # sample_premise="[FEMALE] decided she was finally ready to get a pet."
    # sample_hypothesis="[FEMALE] go to the pet store and buy a pet."
    # sample_update="The pet store is closed."
    
    
    predicted, probs, additional_info = clf_model.predict(sample_premise, sample_hypothesis, sample_update)
    return jsonify({'predicted': predicted,
                'probs': probs,
                'additional_info': additional_info})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1999)