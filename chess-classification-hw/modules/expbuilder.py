import numpy as np
import time
import copy
import json
import os
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
from fastai2.vision.all import (get_image_files, 
                                ImageDataLoaders,
                                RandomResizedCrop,
                                aug_transforms,
                                cnn_learner,
                                L,
                                resnet18,
                                resnet34,
                                resnet50,
                                error_rate,
                                ClassificationInterpretation,
                                flatten_check,
                                accuracy,
                                TensorCategory,
                                Callback,
                                ResizeMethod,
                                Resize,

                                )

from .tfmsutils import MyResizeDeterm

from .trainutils import (piece_class_parse,
                        new_file,
                        stratify_sample,
                        build_dl,
                        subcat_color_acc,
                        subcat_piece_acc
                        
                        )

from .learnutils import (get_cb,
                         TestSetRecorder,
                         learner_add_testset

                        )

'''



'''

def save_learner_logs(learn, 
                      name,
                      log_path='../models/model-logs/',
                      b_msg=False,
                      msg_cols=['test_loss']
                      ):

    valid_recorder = get_cb(learn, 'Recorder')
    test_recorder =  get_cb(learn, 'TestSetRecorder')

    df_valid = pd.DataFrame(valid_recorder.values, 
                            columns = valid_recorder.metric_names[1:-1])

    cols = ['loss']
    cols += [e.name for e in valid_recorder.metrics]
    cols = ['test_' + col for col in cols]

    df_test = pd.DataFrame(test_recorder.values, columns = cols)

    df_valid.to_csv(log_path + name + '_valid.csv', index=False)
    df_test.to_csv(log_path + name + '_test.csv', index=False)

    if b_msg:
        print('Finished with:\n')
        try:
            _cols = [e for e in msg_cols if e in df_valid.columns]
            print(df_valid.iloc[-1:,].loc[_cols])
            _cols = [e for e in msg_cols if e in df_test.columns]
            print(df_test.iloc[-1:,].loc[_cols])
        except:
            print('error printing stats from run')

def save_residual_log(learn,
                      name,
                      log_path = '../models/model-logs/'
                      ):

    preds = learn.get_preds(ds_idx=2, with_loss=True)

    fns =     [e.name for e in learn.dls[2].items]
    labels =  [piece_class_parse(e) for e in fns]

    y_actual = preds[1].tolist()
    y_hat =    torch.argmax(preds[0], dim=1).tolist()
    y_loss =   preds[2].tolist()

    d_preds  = {
        'fn':     fns.copy(),
        'label':  labels.copy(),
        'actual': y_actual.copy(),
        'pred':   y_hat.copy(), 
        'loss':   y_loss.copy(),
        }

    df_preds = pd.DataFrame(d_preds)

    df_preds.to_csv(log_path + name + '_residuals.csv')


def save_learner_pkl( learn,
                      name,
                      pkl_path = '../models/'
                      ):
    old_path = learn.path
    learn.path = Path(pkl_path)
    learn.export(name + '.pkl')
    learn.path = old_path


def save_learner_params(d_params, 
                        name, 
                        log_path='../models/model-logs/'
                        ):
    
    d_params.update({'model_fn': name})

    with open(log_path + name + '_params.json', 'w') as f:
        json.dump(d_params, f)


def save_learner(learn,
                name_base,
                d_params,
                log_path='../models/model-logs/',
                pkl_path='../models/',
                b_msg=False,
                msg_cols = ['test_accuracy'],
                ):
    '''
        outputs files:

            log_path/ 
                <name>_valid.csv    - per epoch metrics on valid
                <name>_test.csv     - per epoch metrics on test
                <name>_residuals.csv - pred, actual, loss on each test set item
                <name>_params.json  - params dict

                <name> = <name_base>-<N>  where <N> is found via `new_file`

            pkl_path/
                <name>.pkl         - exported learner

    '''
    
    # use log_path to find a new_name by adding a number to 
    # the end of name_base. here we use _valid.csv log file to check
    # for the new_name
    fns = os.listdir(log_path)
    new_name = new_file(fns, 
                        prefix=name_base + '_valid',
                        ext='.csv',
                        new_ext=None,  # returns new_name w/o ext
                        )
    
    save_learner_params(d_params, new_name, log_path=log_path)
    
    save_learner_logs(learn, new_name, log_path=log_path, b_msg=b_msg, msg_cols=msg_cols)

    save_residual_log(learn, new_name, log_path=log_path)

    save_learner_pkl(learn, new_name, pkl_path=pkl_path)
    

default_params = {
        '_train_name':              'rf-v1-crops',
        '_test_name':               'test-regulation-2-all',
        '_train_path':              Path('../../../rf-chess-data/cropped_v1/'),
        '_test_path':               Path('../../../other-chess-data/regulation-test-all/'),
        '_model_arch':              resnet50,
        '_fit_one_cycle_epochs':    10,
        '_fine_tune_epochs':        15,
        '_train_seed':              0,
        '_valid_pct':               0.2,
        '_mult':                    1.0,
        '_max_lighting':            0.9,
        '_max_warp':                0.4,
        '_max_rotate':              20.,
        '_resize_method':           ResizeMethod.Pad,
        '_pad_mode':                'reflection',
        '_bs':                      32,
        '_p_lighting':              0.75,
        '_custom_crop':             None,
        '_custom_train_fnames':     None,
        '_custom_train_fnames_args': {},
        
    }

'''
default_params = {
        '_train_name':              'rf-v1-crops',
        '_test_name':               'test-regulation-2-all',
        '_train_path':              ?              
        '_test_path':               ?
        '_model_arch':              resnet50,
        '_fit_one_cycle_epochs':    10,
        '_fine_tune_epochs':        15,
        '_train_seed':              0,
        '_valid_pct':               0.2,
        '_mult':                    1.0,  up to 2.0
        '_max_lighting':            0.9,   up to 0.95
        '_max_warp':                0.4,    
        '_max_rotate':              20.,
        '_resize_method':           ResizeMethod.Pad,  ResizeMethod.Crop, 
        '_pad_mode':                'relection',  'zeros'
        '_bs':                      32,
        '_p_lighting':              0.75,
        '_custom_crop':             None,   'my-top-crop'
        '_custom_train_fnames':     None,   'stratify'
        '_custom_train_fnames_args': {},    {'path':_train_path, n:200, np_seed:42}
    }

    # build args list macro
    msg = ""
    for k in default_params.keys():
        msg += f"{k} = params.get('{k}')\n"
    print(msg)

'''


def run_exp(params, 
            name_base,
            b_subcat_metrics=True,
            b_msg=True,
            msg_cols=['valid_loss', 'accuracy', 'test_loss', 'test_accuracy']
            ):
    '''
        input:
            params      - (dict) {param-names[i] param-value[i]}
            name_base   - (str)  what to name the files output
        
        return value: 
            None

        outputs: 
            several files, see `save_learner` doc

        TODO - 
            could do params=params,... b_msg=True, **params
            (instead of manually unpacking)

    '''

    ## Build local parameters ----
    
    _train_name = params.get('_train_name')
    _test_name = params.get('_test_name')
    _train_path = params.get('_train_path')
    _test_path = params.get('_test_path')
    _model_arch = params.get('_model_arch')
    _fit_one_cycle_epochs = params.get('_fit_one_cycle_epochs')
    _fine_tune_epochs = params.get('_fine_tune_epochs')
    _train_seed = params.get('_train_seed')
    _valid_pct = params.get('_valid_pct')
    _mult = params.get('_mult')
    _max_lighting = params.get('_max_lighting')
    _max_warp = params.get('_max_warp')
    _max_rotate = params.get('_max_rotate')
    _resize_method = params.get('_resize_method')
    _pad_mode = params.get('_pad_mode')
    _bs = params.get('_bs')
    _p_lighting = params.get('_p_lighting')
    _custom_crop = params.get('_custom_crop')
    _custom_train_fnames = params.get('_custom_train_fnames')
    _custom_train_fnames_args = params.get('_custom_train_fnames_args')
    
    # print(type(_custom_train_fnames_args))
    # print(type(_custom_train_fnames_args['np_seed']))

    # obj's -> str (where nec.) to dump params as json at end
    save_params = params.copy()
    save_params['_model_arch'] = save_params['_model_arch'].__name__
    save_params['_train_path'] = str(save_params['_train_path'])
    save_params['_test_path'] =  str(save_params['_test_path'])
    save_params['_custom_train_fnames_args'] =  str(save_params['_custom_train_fnames_args'])
    

    ## Build Data -----------

    train_fnames = get_image_files(_train_path)
    
    if _custom_train_fnames == 'stratify':
        train_fnames =  stratify_sample(**_custom_train_fnames_args)
        print(len(train_fnames))
    
    
    
    Crop = Resize(128, _resize_method, pad_mode=_pad_mode)
    
    if _custom_crop == 'my-top-crop':
        Crop = MyResizeDeterm(128, _resize_method, pad_mode=_pad_mode)

    Augs = aug_transforms(mult=_mult, 
                          max_lighting=_max_lighting,
                          p_lighting=_p_lighting, 
                          max_warp=_max_warp,
                          max_rotate=_max_rotate,
                         )
    
    train_dl = ImageDataLoaders.from_name_func(
                    _train_path, 
                    train_fnames,
                    valid_pct=_valid_pct, 
                    seed=_train_seed,
                    label_func=piece_class_parse, 
                    item_tfms=Crop,
                    # batch_tfms=Augs,
                    bs=_bs,
                    )
    # print(train_dl.train.items)

    test_dl = build_dl(_test_path)

    ## Build Learner -------

    if b_subcat_metrics:
        learn_metrics = [accuracy, subcat_color_acc, subcat_piece_acc]
    else:
        learn_metrics = [accuracy]
    
    learn = cnn_learner(train_dl, _model_arch, metrics=learn_metrics)

    learner_add_testset(learn, test_dl)

    learn.add_cb(TestSetRecorder())

    ## Fit Learner ------------

    if b_msg: print('starting training...')

    t0 = time.time()

    with learn.no_logging(): learn.fit_one_cycle(_fit_one_cycle_epochs)

    with learn.no_logging(): learn.fine_tune(_fine_tune_epochs)

    t1 = time.time()

    ## Metrics + Save ---------

    save_learner(learn, 
                name_base=name_base, 
                d_params=save_params,
                b_msg=b_msg,
                msg_cols=msg_cols,
                )

    if b_msg:
        print(f'finished in {round(t1 - t0, 2)} secs\n')


    