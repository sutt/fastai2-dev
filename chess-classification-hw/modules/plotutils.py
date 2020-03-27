import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def plot_many_imgs(imgs, titles=None):
    
    n = len(imgs)
    if n > 80:
        print(f"Images of len {n} restricted to 80")
        n = 80
        
    if titles is None:
        titles = list(range(n))

    fig, ax = plt.subplots(nrows=n, ncols=1, figsize=(6, n*3))

    for _i, (_img, _title) in enumerate(zip(imgs, titles)):
        
        ax[_i].imshow(_img)
        ax[_i].axis('off')        
        ax[_i].title.set_text(str(_title))
        
        
def load_many_imgs(d, N=10):
    
    sel_fns = np.random.choice(os.listdir(d),size=N).tolist()
    sel_fns = [d / e for e in sel_fns]

    imgs = []
    for f in sel_fns:
        imgs.append(np.array(Image.open(f)))

    print(f"loaded {len(imgs)} images")
    return imgs