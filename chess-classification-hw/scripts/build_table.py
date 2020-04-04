import numpy as np
import os, sys, json
import PIL
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# img_dir = '../../rf-chess-data/roboflow/export/'
# img_fns = os.listdir(img_dir)
# output_dir = 'data/rf-annotate/'
# annotate_dir = '../../rf-chess-data/roboflow/export/'
# annotate_fn = '_annotations.coco.json'

annotate_dir = '../../../other-chess-data/regulation-pieces-1/originals/data/'
annotate_fn = 'instances.json'

output_dir = '../data/other-annotate-rp1/'
output_fn = output_dir + 'data_v1.csv'


with open(Path(annotate_dir, annotate_fn), 'r') as f:
    d_annotate = json.load(f)

df_annotate   = pd.DataFrame(d_annotate['annotations'])
df_images     = pd.DataFrame(d_annotate['images'])
df_categories = pd.DataFrame(d_annotate['categories'])

# merges + drop unnec cols
df_join = pd.merge(df_annotate, 
                   df_images,
                   how='left',
                   left_on='image_id',
                   right_on='id')

drop_cols = ["id_x","id_y", "license", 
             "segmentation", "iscrowd", "date_captured"]

for drop_col in drop_cols:
    try:df_join = df_join.drop([drop_col], axis=1)
    except:pass

df_join = pd.merge(df_join,
                    df_categories,
                    how='left',
                    left_on='category_id',
                    right_on='id'
                   )

df_join = df_join.drop(["id", "supercategory"], axis=1)
df_join = df_join.rename(mapper={'name': 'category_full_name'}, axis=1)

# add extra cols
df_join['annotate_id'] = df_join.index

df_join['coord_tl'] = df_join['bbox'].map(lambda x: [x[0], x[1]])
df_join['coord_br'] = df_join['bbox'].map(lambda x: [x[0] + x[2], x[1] + x[3]])

df_join['category_color_name'] = df_join['category_full_name'].map(lambda x: x.split('-')[0])
df_join['category_piece_name'] = df_join['category_full_name'].map(lambda x: x.split('-')[1])

df_join.to_csv(output_fn, index=False)