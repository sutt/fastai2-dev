import sys, time
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from ast import literal_eval

def load_np_img(img_fn):
    img_pil = Image.open(Path(img_dir, img_fn))
    img_np = np.array(img_pil)
    return img_np

def crop_img(img, bbox):
    lx, ly, w, h = bbox
    return img[ly:ly+h,lx:lx+w ,:] 

# cloud
# img_dir = Path('../../rf-chess-data/export/')

# locally
img_dir = Path('../../../other-chess-data/regulation-pieces-3/originals/')
save_dir = Path('../../../other-chess-data/regulation-pieces-3/crops/')
meta_df = '../data/other-annotate-rp1/data_reg3_v1.csv'

# other
# img_dir = Path('../../../other-chess-data/regulation-pieces-1/originals/')
# save_dir = Path('../../../other-chess-data/regulation-pieces-1/crops')
# meta_df = '../data/other-annotate-rp1/data_v1.csv'



df = pd.read_csv(meta_df)
df['bbox'] = df['bbox'].map(lambda x: literal_eval(x))

if len(sys.argv) > 1:
    print('in here')
    df = df.iloc[:4,:]

t0 = time.time()

for _i in range(len(df)):
    
    _row = df.iloc[_i,:]
    
    _img = load_np_img(_row['file_name'])
    
    _imgcropped = crop_img(_img, _row['bbox'])
    
    _annid = _row['annotate_id']
    _catid = _row['category_full_name']

    title =  str(_annid).zfill(5)
    title += "_"
    title += _catid
    title += ".jpg"

    _imgpil = Image.fromarray(_imgcropped)
    
    _imgpil.save(save_dir / title)

print(f'finished in: {round(time.time() - t0, 0)}')
