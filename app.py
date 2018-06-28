from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from flask import Flask, redirect, url_for, request, render_template, json
from flask_cors import CORS
from gevent.pywsgi import WSGIServer


from age import AgeEstimator
from res50 import RN50

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
CORS(app)
port = 8080

print("Server running on {} port", port)

@app.route('/', methods=['GET'])
def index():
    return json.dumps({
        'app': 'recon',
        'version': '0.1'
        })


@app.route('/res50', methods=['GET', 'POST'])
def Res50Upload():
    if request.method == 'POST':
        f = request.files['file']
        result = json.dumps(RN50(f), cls=NumpyEncoder)
        return result
    return None

@app.route('/ageGender', methods=['GET', 'POST'])
def AgeGenderUpload():
    if request.method == 'POST':
        f = request.files['file']
        result = json.dumps(AgeEstimator(f), cls=NumpyEncoder)
        return result
    return None


if __name__ == '__main__':
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()
