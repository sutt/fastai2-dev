from fastai2.vision.all import *
import time

def piece_class_parse(fn): 
    fn = fn.split('_')[1]
    fn = fn.split('.')[0]
    return fn

# learn2 = load_learner('../models/chess1.pkl')
learn = load_learner('../models/stadard-piece-2.pkl')

img_fns = [ Path('../img-tmp/white-pawn-2.jpg'),
            Path('../img-tmp/white-bishop-2.jpg'),
            Path('../img-tmp/black-knight-2.jpg'),
            Path('../img-tmp/white-bishop-3.jpg'),
            Path('../img-tmp/black-knight-3.jpg'),]

print('sleeping 1')
time.sleep(1)

path = img_fns[0]
print(path)
ret1 = learn.predict(path)
# ret2 = learn2.predict(path)
# print(ret1) '\n-----\n', ret2
print(ret1)

print('sleeping 1')
time.sleep(1)

t0 = time.time()
for _ in range(10):
    learn.predict(img_fns[1])
t1 = time.time()

print(f"10 predictions in {round(t1 - t0,1)} secs")