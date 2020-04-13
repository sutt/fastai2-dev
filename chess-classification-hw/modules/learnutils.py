import numpy as np
import time
import copy
import os
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
# from .trainutils import ()
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
    def __init__(self, ds_idx=2, **kwargs):
        self.values = []
        self.ds_idx = ds_idx
        self.counter = 0
        self.skip_counter = None
    
    def after_epoch(self):
        self.counter += 1
        if self.skip_counter is not None:
            if self.counter < self.skip_counter: return
        old_log = self.recorder.log.copy()
        self.learn._do_epoch_validate(ds_idx=self.ds_idx, dl=None)
        self.values.append(self.recorder.log[len(old_log):])
        self.recorder.log = old_log


def learner_add_testset(learn, test_dl, b_cuda=False):
    new_dl = DataLoaders(learn.dls[0], learn.dls[1], test_dl.train)
    if b_cuda: new_dl.cuda()
    learn.dls = new_dl


def get_cb_index(learn, cb_name):
    return [i for i, e in enumerate(list(learn.cbs)) 
            if e.__class__.__name__ == cb_name][0]

def get_cb(learn, cb_name):
    return learn.cbs[get_cb_index(learn, cb_name)]

def ignore_first_testset_callback(learn):
    test_recorder_ind =  get_cb_index(learn, 'TestSetRecorder')
    learn.cbs[test_recorder_ind].counter = -1
    learn.cbs[test_recorder_ind].skip_counter = 1