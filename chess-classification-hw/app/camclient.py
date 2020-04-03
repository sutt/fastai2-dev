import threading
import queue
import requests
import json
import cv2
import imutils
import time
import argparse
from modules.clienthelpers import (ping, draw, annotate_class, annotate_think)


'''
    quickstart:

    launch flask server (terminal1):
        /app> wsl
        /app> conda activate fastai2
        /app> python predserver.py  [--log] [--debug] [--model <model_path>]

    run video client (terminal2):
        /app> conda activate base
        /app> python camclient.py  [--log] [--debug] [--framemod <int>] [--camnum <int>]

    note on deployment:
        model predcition server run's in the `fastai2` 
        environment in WSL, put this can't use hardware like
        cv2.VideoCapture, so we run that in Window's in the 
        `base` conda environment (which has cv2 library)

    use browser to view prediction of last snap at: 
        localhost:5000/log

    CLI args:

    --thinkoff  (bool flag) - turn off printing progressbar at bottom
    --camnum    (int)       - 0:front facing, 1:rear facing
    --framemod  (int)       - how many frames to read before sending request
    --debug     (bool flag) - print to console (for timing issues)

'''

ap = argparse.ArgumentParser()
ap.add_argument("--thinkoff",  action="store_true", default=False)
ap.add_argument("--camnum", type=str, default="1")
ap.add_argument("--framemod", type=str, default="120")
ap.add_argument("--debug",  action="store_true", default=False)
args = vars(ap.parse_args())

                  # ((x0,  y0),  (x1, y1))
frame_rect =        ((200, 100), (400,300))
                   
# frame_rect =        ((200,50), (400, 400))


cam_num             = int(args['camnum'])
framemod            = int(args['framemod'])
b_think_annotate    = not(args['thinkoff'])

b_debug             = args['debug']

q                   = queue.Queue()
counter             = -10
b_starting_thread   = True
response_text       = None

b_reset_think       = True
b_snap              = False
think_counter       = 0
mod_think_counter   = framemod // 15

camera = cv2.VideoCapture(cam_num)

while(camera.isOpened()):

    grabbed, frame = camera.read()
    frame = imutils.resize(frame, width=600)
    frame = draw(frame,frame_rect)
    
    try:
        response_text = q.get(False)  
        b_reset_think = True
        t1 = time.time()
        if b_debug: print(f"~~~~ server response time: {round(t1 - t0,0)}")
    except queue.Empty:
        pass
    except:
        response_text = "idk? exception in get"

    if (response_text is None) and b_starting_thread:
        response_text = 'starting...'
        b_starting_thread = False
        
    frame = annotate_class(frame, response_text)

    if b_think_annotate:

        frame = annotate_think( frame,
                                b_snap, 
                                int(think_counter // mod_think_counter))
    
    cv2.imshow("RealTime Image Classifier", frame)

    if counter % framemod == 0:

        if b_debug: print('snap')
        b_snap = True
        think_counter = 0
        t0 = time.time()
        
        ((x0,  y0),  (x1, y1)) = frame_rect

        t = threading.Thread(target = ping, 
                             args = (
                                 frame[y0:y1, x0:x1,:],
                                 q,
                                 b_debug
                                 ) )
        t.daemon = True
        t.start()
    else:
        think_counter += 1
        b_snap = False

    counter += 1
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()



