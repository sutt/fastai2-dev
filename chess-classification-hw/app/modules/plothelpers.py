import sys
import time
import json
import io
import random
from PIL import Image
from flask import Response
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def create_figure():

    figsize= (8, 4)
    pic_fn = 'data/pic.jpg'
    
    with open('data/pred.json') as f:
        d = json.load(f)

    pred_class =    d['pred-class-name']
    y =             d['output-layer-activations']
    x_titles =      d['class-titles']
    last_saved =    time.time() - d['time-saved']

    assert len(y) == len(x_titles)

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].barh(x_titles, y)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_title(f'predicted: {pred_class}')
    
    ax[1].imshow(Image.open(pic_fn))
    ax[1].axis('off')
    ax[1].set_title(f'last updated: {round(last_saved,2)} secs ago')

    plt.tight_layout()
    
    return fig

def response_plot():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')