from flask import Flask, request, Response, jsonify
import jsonpickle
import numpy as np
import cv2
from fastai2.vision.all import *
import time
import sys
import json

'''

useful discussion of this flask as image server:
https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594


'''

model_fn = '../models/stadard-piece-2.pkl'

app = Flask(__name__)

def piece_class_parse(fn): 
    fn = fn.split('_')[1]
    fn = fn.split('.')[0]
    return fn

learn = load_learner(model_fn)


@app.route('/pred', methods=['POST'])
def pred():
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

if __name__ == "__main__":
    app.run(debug=True)