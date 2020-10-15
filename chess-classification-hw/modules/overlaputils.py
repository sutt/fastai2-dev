import numpy as np
import os, sys, json
import PIL
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')


def foo(
    df,
    cat_name='black-queen',
    crop_dir='../../../rf-chess-data/cropped_verify/',
    max_cols=15,
    thresh=10,
    b_ret=False,
    ):
    
    bq = df[df['category_full_name'] == cat_name]

    corners = bq['bbox'].map(lambda x: (x[0], x[1])).tolist()

    def dist(a,b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    twins = [ [ i2 for i2, c2 in enumerate(corners) 
                    if (i2 != i1) and (dist(c1, c2) < thresh)] 
                for i1, c1 in enumerate(corners)]

    def sorter(i,v):
        x = list(set([i] + v))
        x.sort()
        return x

    twin_sets = [sorter(i,v) for i,v in enumerate(twins)]

    bq['twin_sets'] = twin_sets
    bq['twin_sets_obj'] = bq['twin_sets'].map(str)

    def make_set(s1, s2):
        
        shared_1 = [e for e in s1 if e in s2]
        
        if len(shared_1) < 0.5 * len(s1):
            return s1
        
        ret = list(set(s1).union(s2))
        ret.sort()
        return ret

    def big_set(*args):
        ret = [(len(e), e) for e in args]
        ret.sort(reverse=True, key=lambda x: x[0])
        return ret[0][1]

    list_ts = bq['twin_sets'].tolist()
    out = list_ts.copy()
    for _ in range(10):
        out = [ big_set(*[make_set(ts1, ts2) for ts2 in out]) for ts1 in out]

    bq['twin_set_cluster'] = out
    bq['twin_set_cluster_obj'] = bq['twin_set_cluster'].map(str)

    if b_ret: return bq

    amts = bq.groupby('twin_set_cluster_obj').agg('count').iloc[:,1]

    max_show = 99

    matched = amts[amts > 1]
    unmatched = amts[amts == 1]

    iters = matched.sort_values(ascending=False).index[:max_show]

    cols = min(max_cols, max([e.count(',') + 1 for e in iters]))
    rows = len(iters) + ((len(unmatched) // cols)+1)

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 10 * rows));

    for irow, twin_code in enumerate(iters):

        tmp = bq[bq['twin_set_cluster_obj'] == twin_code]

        ifns = [str(e).zfill(5) for e in tmp.index]

        clip_fns = [[e for e in os.listdir(crop_dir) if ifn in e][0] for ifn in ifns]
        clips = [PIL.Image.open(crop_dir + fn) for fn in clip_fns]
        
        for icol, pil in enumerate(clips):
            if icol >= max_cols - 1: break
            ax[irow][icol].imshow(np.array(pil))
            

    # second half fill in unqiue images -----

    irow += 1
    counter = 0
    for idata, twin_code in enumerate(unmatched.index[:max_show]):

        tmp = bq[bq['twin_set_cluster_obj'] == twin_code]

        ifns = [str(e).zfill(5) for e in tmp.index]

        clip_fns = [[e for e in os.listdir(crop_dir) if ifn in e][0] for ifn in ifns]
        clips = [PIL.Image.open(crop_dir + fn) for fn in clip_fns]

        for pil in clips:
            counter_row = counter // cols
            counter_col = counter % cols
            ax[irow + counter_row][counter_col].imshow(np.array(pil))
            counter += 1
        
    plt.show()