from flask import Flask, request, Response, jsonify
import jsonpickle
import numpy as np
import cv2
import time
import sys
import json
import argparse
from pathlib import Path
from fastai2.vision.all import *

from appmodules.plothelpers import response_plot

# little hack here so `load_learner()`
# can find ../modules/trainutils `piece_class_parse` to load 
# the model. Also why we call the app's pkg dir `appmodules`
sys.path.insert(0, '..')

'''
    activate fastai2 conda-env in wsl before running server:

        --model <path/to/model.pkl>     - change model for prediction
        --debug                         - turn on printing ton console
        --log                           - turn on saving to app/data/ of:
                                          (last snapshot, last model pred,
                                           model class names, last time saved)

    useful discussion of this flask as image server:
    https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594

'''

ap = argparse.ArgumentParser()
# ap.add_argument("--model", type=str, default="base-learner-13-fit25.pkl") 
ap.add_argument("--model", type=str, default="base-learner-6.pkl")  # "expmod-b-1.pkl"
ap.add_argument("--debug", action="store_true", default=False)
ap.add_argument("--log", action="store_true", default=False)
args = vars(ap.parse_args())

model_dir   = Path('../models/') 
model_fn    = model_dir / args["model"]

debug = args['debug']
store = args['log']

app = Flask(__name__)

def piece_class_parse(fn): 
    fn = fn.split('_')[1]
    fn = fn.split('.')[0]
    return fn

learn = load_learner(model_fn)


@app.route('/pred', methods=['POST'])
def pred():
    
    if debug:
        t0 = time.time()
        print('\nin pred ----------------------------------')
        sys.stdout.flush()
    
    r = request

    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    ret1 = learn.predict(img)

    d = {'pred-class-name': ret1[0],
         'pred-class-index': ret1[1].tolist(),
         'output-layer-activations': ret1[2].tolist(),
         'class-titles': list(learn.dls.vocab),
         'time-saved': time.time(),
    }

    if log:
        
        with open('data/pic.jpg', 'wb') as f:
            f.write(r.data)

        with open('data/pred.json', 'w') as f:
            json.dump(d, f)

    if debug:
        print(ret1)
        print(f'end of pred, time: {round(time.time() - t0,2)} ----------------------')
        sys.stdout.flush()

    return jsonify(d)


@app.route('/log')
def log():
    return response_plot()


if __name__ == "__main__":
    app.run(debug=True)