'''
useful discussion of this flask as image server:
https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594

'''
from flask import Flask, request, Response, jsonify
import jsonpickle
import numpy as np
import cv2
from fastai2.vision.all import *
import time
import sys
import json

def piece_class_parse(fn): 
    fn = fn.split('_')[1]
    fn = fn.split('.')[0]
    return fn

# learn2 = load_learner('../models/chess1.pkl')
learn = load_learner('../models/stadard-piece-2.pkl')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/img/')
def img():
    pass

@app.route('/pred', methods=['POST'])
def pred():
    r = request

    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    ret1 = learn.predict(img)
    print(ret1)
    sys.stdout.flush()

    return str(ret1)

@app.route('/pred2', methods=['POST'])
def pred2():
    r = request

    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    ret1 = learn.predict(img)
    print(ret1)
    sys.stdout.flush()

    d = {'pred-class-name': ret1[0],
         'pred-class-index': ret1[1].tolist(),
         'output-layer-activations': ret1[2].tolist(),
    }

    return jsonify(d)





@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
    app.run(debug=True)