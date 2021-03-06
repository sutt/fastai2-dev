import numpy as np
import time
import copy
import os
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
# from .trainutils import ()
from fastcore.transform import Pipeline
from fastai2.vision.all import (get_image_files, 
                                ImageDataLoaders,
                                RandomResizedCrop,
                                aug_transforms,
                                cnn_learner,
                                L,
                                resnet18,
                                error_rate,
                                ClassificationInterpretation,
                                flatten_check,
                                accuracy,
                                TensorCategory,
                                Callback,
                                DataLoaders,
                                
                                )

class TestSetRecorder(Callback):
    def __init__(self, ds_idx=2, b_logger=False, **kwargs):
        self.values = []
        self.ds_idx = ds_idx
        self.counter = 0
        self.skip_counter = None
        self.b_logger = b_logger
    
    def after_epoch(self):
        self.counter += 1
        if self.skip_counter is not None:
            if self.counter < self.skip_counter: return
        old_log = self.recorder.log.copy()
        self.learn._do_epoch_validate(ds_idx=self.ds_idx, dl=None)
        self.values.append(self.recorder.log[len(old_log):])
        self.recorder.log = old_log
        if self.b_logger:
            self.log = self.recorder.log[len(old_log):]
            self.logger(self.log)


def learner_add_testset(learn, test_dl, b_cuda=False):
    new_dl = DataLoaders(learn.dls[0], learn.dls[1], test_dl.train)
    if b_cuda: new_dl.cuda()
    learn.dls = new_dl
    

def learner_add_testset_2(learn, test_path, b_cuda=False):
    
    built_test = learn.dls.test_dl(get_image_files(test_path), 
                                with_labels=True)

    new_dl = DataLoaders(learn.dls[0], learn.dls[1], built_test)
    if b_cuda: new_dl.cuda()
    learn.dls = new_dl

def learner_rm_norm(learn):
    ''' remove Normalize from after_batch in all dl's '''

    pipe0 = learn.dls[0].after_batch

    pipe0_prime = Pipeline()
    pipe0_prime.add(pipe0[0])
    pipe0_prime.add(pipe0[1])
    pipe0_prime.add(pipe0[2])

    for i, _ in enumerate(learn.dls):
        learn.dls[i].after_batch = pipe0_prime


def get_cb_index(learn, cb_name):
    return [i for i, e in enumerate(list(learn.cbs)) 
            if e.__class__.__name__ == cb_name][0]

def get_cb(learn, cb_name):
    return learn.cbs[get_cb_index(learn, cb_name)]

def ignore_first_testset_callback(learn):
    test_recorder_ind =  get_cb_index(learn, 'TestSetRecorder')
    learn.cbs[test_recorder_ind].counter = -1
    learn.cbs[test_recorder_ind].skip_counter = 1