import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def load_many_imgs(d, N=10):
    
    sel_fns = np.random.choice(os.listdir(d),size=N).tolist()
    sel_fns = [d / e for e in sel_fns]

    imgs = []
    for f in sel_fns:
        imgs.append(np.array(Image.open(f)))

    print(f"loaded {len(imgs)} images")
    return imgs
