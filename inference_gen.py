from flask import Flask
from flask import jsonify
from flask import request

import json

#MODEL RELATED
from defeasible_gen_model import *

app = Flask(__name__)

print("Loading Defeasible GEN ATOMIC")
gen_model = DefeasibleGenModel(data_type="atomic")

#Demo
# @app.route('/')
# def hello():
#     return 'Hello World!'

#Receives Parameter from gunicorn
# def load_app(gpu_num):
#     cfg = load_app_config(cfg_file)
#     return my_app(cfg)


#Only accept POST Request
@app.route('/predict_gen', methods=['POST'])
def predict_gen():
    if request.method == 'POST':
        # Get received JSON
        recv_json=request.json  
        print(recv_json)

        premise=recv_json["premise_input"]
        hypothesis=recv_json["hyp_input"]

    sample_premise="[MALE] has a very important math test next week."
    sample_hypothesis="[MALE] get a good grade on the test."

    val,additional_info = gen_model.generate(sample_premise,sample_hypothesis)
    return jsonify({'test_val': val,
                'additional_info': additional_info})