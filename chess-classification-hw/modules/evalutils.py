from fastai2.vision.all import load_learner, get_image_files
import torch
from pathlib import Path
import json, os, sys
import pandas as pd
sys.path.append('..')
from modules.trainutils import piece_class_parse
from modules.trainutils import my_metrics
from matplotlib import pyplot as plt

def get_dir_nums(base_name=None, log_dir='../models/model-logs/'):
    fns = os.listdir(log_dir)
    if base_name is not None: fns =  [fn for fn in fns if base_name in fn ]
    nums = [e.split('-')[1].split('_')[0] for e in fns]
    nums = list(set([int(e) for e in nums]))
    return nums

def param_diffs(df_params):
    s = df_params.T.nunique(dropna=False)
    return s[s > 1]

def get_tbls(name, nums, log_dir='../models/model-logs/'):

    log_dir = Path(log_dir)
    
    df_metrics = pd.DataFrame()
    df_resid = pd.DataFrame()
    df_params = pd.DataFrame()

    for i in nums:

        ## read in all four tables ------
        
        base_name = name + '-' + str(i)

        _df_valid = pd.read_csv(log_dir / (base_name + "_valid.csv"))                       
        _df_test  = pd.read_csv(log_dir / (base_name + "_test.csv"))
        
        _df_resid  = pd.read_csv(log_dir / (base_name + "_residuals.csv"))

        with open(log_dir / (base_name + "_params.json")) as f:
            _d_params = json.load(f)
            
        ## concat + format df_valid, df_test -----
        _df_metrics = pd.concat((_df_valid, _df_test), axis=1)
        
        _df_metrics['epoch'] = _df_test.index
        _df_metrics['exp_name'] = base_name
        
        cols =  _df_metrics.columns[-2:].tolist()
        cols += _df_metrics.columns[:-2].tolist()
        
        _df_metrics = _df_metrics[cols]
        
        ## format df_resid ---------
        _df_resid.drop(columns=['Unnamed: 0'], inplace=True)
        
        _df_resid['correct'] = _df_resid['actual'] == _df_resid['label']
        _df_resid = _df_resid.rename(columns={
                            'pred': base_name + '_pred',
                            'loss': base_name + '_loss',
                            'correct': base_name + '_correct'})
        
        ## format df_params other tables
        _df_params = pd.DataFrame(_d_params, index=[base_name])
        
        
        ## concat to total tables ---------
        df_metrics = pd.concat((df_metrics, _df_metrics), axis=0)
        
        if 'label' not in df_resid.columns:
            df_resid = _df_resid.copy()
        else:
            cols = _df_resid.columns.difference(df_resid.columns).tolist()
            cols += ['fn']
            df_resid = pd.merge(df_resid, _df_resid[cols], 
                                how='outer', on='fn')
            
        df_params = pd.concat((df_params,_df_params ), axis=0)
    
    return df_metrics, df_resid, df_params