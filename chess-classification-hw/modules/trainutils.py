import numpy as np
import time
import copy
import os
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
from fastai2.vision.all import (get_image_files, 
                                ImageDataLoaders,
                                RandomResizedCrop,
                                aug_transforms,
                                cnn_learner,
                                L,
                                resnet18,
                                error_rate,
                                ClassificationInterpretation,

                                )



def piece_class_parse(fn): 
    ''' 01415_white-rook.jpg -> white-rook '''
    fn = fn.split('_')[1]
    fn = fn.split('.')[0]
    return fn


def only_class_parse(fn):
    ''' 01415_white-rook.jpg -> rook '''
    fn = fn.split('_')[1]
    fn = fn.split('-')[1]
    fn = fn.split('.')[0]
    return fn

def filter_piece_color(path, color='white'):
    fns = get_image_files(path)
    color_pieces = [i for i,v in enumerate(fns)
                    if color in v.name]
    return L([fns[i] for i in color_pieces])


def stratify_sample(path, n=100, np_seed=None, color=None):
    
    fns = get_image_files(path)

    if color is not None:
        fns = filter_piece_color(path, color=color)
    
    if np_seed is not None:
        np.random.seed(np_seed)
    
    classes = [piece_class_parse(e.name) for e in fns]
    all_classes = list(set(classes))
    n_per_class = n // len(all_classes)
    
    rand_inds = []
    for _class in all_classes:
        
        _classes = [i for i,v in enumerate(classes) if v == _class]
        
        rand_inds.extend(np.random.choice(_classes, n_per_class))
        
    return L([fns[i] for i in rand_inds])

def silent_learner(learn):
    '''remove cbs: Recorder,ProgressCallback '''
    learn.cbs.pop(2)
    learn.cbs.pop(1)


def my_metrics(learn, dl):

    preds_test = learn.get_preds(dl=dl, with_loss=True)

    y_actual = preds_test[1].tolist()
    y_hat = torch.argmax(preds_test[0], dim=1).tolist()
    y_loss = preds_test[2].tolist()

    acc = accuracy_score(y_actual, y_hat)

    loss = sum(y_loss) / len(dl.items)
    
    return loss, acc

def my_test_metrics(learn, test_path):
    
    test_dl = learn.dls.test_dl(get_image_files(test_path), 
                                with_labels=True)
    
    return my_metrics(learn, test_dl)


def show_cf(learn, dl):
    interp = ClassificationInterpretation.from_learner(learn, dl=dl)
    interp.plot_confusion_matrix()
    return interp

def my_export(learn, model_fn='tmp-model.pkl'):
    ' to verify run: !ls ../models -sh '
    old_path = learn.path
    learn.path = Path('../models')
    learn.export(model_fn)
    learn.path = old_path

def new_file(fns, prefix='moda', ext='.pkl'):
    '''
        return a new file name with:
            same prefix,
            new "-<num>"
            same ext
    '''
    elems = [fn for fn in fns 
             if fn[:len(prefix)] == prefix]
    
    elems = [e for e in elems
             if e[-len(ext):] == ext]
    
    elems = [e.split('.')[0] for e in elems]
    
    def get_num(s):
        try: 
            num = s.split('-')[-1] 
            num = int(num)
            return num
        except: 
            return -1
        
    elems = [get_num(e) for e in elems]
    
    if len(elems) == 0:
        new_num = 0
    else:
        new_num = max(elems) + 1
    
    new_fn = prefix + '-' + str(new_num) + '.pkl'
    
    return new_fn
    

def my_export_new(learn, fn_dir = '../models', prefix='expmod-a', ext='.pkl',
                  print_check=False):

    fns = os.listdir(fn_dir)
    new_fn = new_file(fns, prefix=prefix, ext=ext)
    my_export(learn, model_fn=new_fn)

    if print_check:
        the_file = [fn for fn in os.listdir(fn_dir) if fn == new_fn]
        if len(the_file) == 0:
            print(f'no new file {new_fn} found in {fn_dir}')
        else:
            print(f'found new file {new_fn} in {fn_dir}')

    return new_fn


def my_acc(learn, test):

    preds_train = [learn.predict(item) for item in learn.dls.train.items]
    # preds_train = [learn.predict(item[1]) for item in learn.dls.dataset]
    y_hat = [e[1].tolist() for e in preds_train]
    y_actual = [e[1].tolist() for e in learn.dls.dataset]
    acc_t = accuracy_score(y_actual, y_hat)

    preds_test = [learn.predict(item) for item in test.train.items]    
    y_hat = [e[1].tolist() for e in preds_test]
    y_actual = [e[1].tolist() for e in test.dataset]
    acc_v = accuracy_score(y_actual, y_hat)
    
    return acc_t, acc_v


def build_dl(path, n=None, seed=None):

    if n is None:
        
        dl = ImageDataLoaders.from_name_func(
                    path, 
                    get_image_files(path),
                    valid_pct=0.0, 
                    seed=seed,      # randomSplitter has no effect
                    label_func=piece_class_parse, 
                    item_tfms=RandomResizedCrop(128, min_scale=0.5),
                    batch_tfms=aug_transforms(),
                    )

    else:

        dl = ImageDataLoaders.from_name_func(
                    path, 
                    stratify_sample(path, n=n, np_seed=seed),
                    valid_pct=0.0, 
                    seed=seed,      # randomSplitter has no effect
                    label_func=piece_class_parse, 
                    item_tfms=RandomResizedCrop(128, min_scale=0.5),
                    batch_tfms=aug_transforms(),
                    )
    return dl


def learn_by_epoch( learn,
                    train,
                    test,
                    epochs=10,
                    b_log=True,
                    ):

    time_tracker = []
    acct_tracker = []
    accv_tracker = []
    
    for epoch in range(epochs):

        t0 = time.time()

        learn.fine_tune(1)

        t1 = time.time()
        
        t = t1 - t0
        acc_t, acc_v = my_acc(learn, test)
        
        time_tracker.append(t)
        acct_tracker.append(acc_t)
        accv_tracker.append(acc_v)
        
        def ff(x,d=3, n=5):
            return str(round(x,d)).rjust(n)
        
        if b_log:
            print(f"epoch: {ff(epoch,2)} | acc_t: {ff(acc_t)} | acc_v: {ff(acc_v)} | time: {ff(t)}")

    return  {  'acc_t': acct_tracker.copy(),
               'acc_v': accv_tracker.copy(),
               'time': time_tracker.copy(), 
            }

def init_trainer(path,
                 test,
                 train_n=100,
                 train_seed=None,
                 epochs=10,
                 b_log=True
                ):

    d = {'train_seed': train_seed,
        'train_n': train_n}

    train = build_dl(path, n=train_n, seed=train_seed)
    
    learn = cnn_learner(train, resnet18, metrics=error_rate)
    
    learn.cbs = L([learn.cbs[0]])

    if b_log:
        print(f"\ntraining {train_n} samples from seed {train_seed} ########\n")
    
    d_metrics = learn_by_epoch(learn, train, test, epochs=epochs, b_log=b_log)

    d.update(d_metrics.copy())
    
    return d


def grid_exp(path,
             d_exps = [
                        {'train_n': 100, 'train_seed': None},
                        {'train_n': 100, 'train_seed': None},
                      ],
             test_n=100,
             test_seed=42,
             epochs=10,
            ):
    
    d = {'test': {'test_seed':  test_seed,
                  'test_n':     test_n,
                 },
        }

    test = build_dl(path, n=test_n, seed=test_seed)
    
    exp_list = []

    for _exp in d_exps:
        
        d_metrics = init_trainer(path, test, epochs=epochs, **_exp)

        exp_list.append(d_metrics.copy())
    
    d['exps'] = exp_list.copy()

    return d

def foo():
    return 2

if __name__ == "__main__":
    print('we guud')