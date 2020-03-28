import threading
import queue
import requests
import json
import cv2
import imutils
import time
import argparse

'''
    quickstart:

    launch flask server (terminal1):
        /app> wsl
        /app> conda activate fastai2
        /app> python flask.py

    run video client (terminal2):
        /app> conda activate bae
        /app> python

    note on deployment:
        model predcition server run's in the `fastai2` 
        environment in WSL, put this can't use hardware like
        cv2.VideoCapture, so we run that in Window's in the 
        `base` conda environment (which has cv2 library)
        

'''

ap = argparse.ArgumentParser()
# ap.add_argument("--file", type=str, default="")
ap.add_argument("--thinkoff",  action="store_true", default=False)
args = vars(ap.parse_args())


def ping(img, q):
    
    addr = 'http://localhost:5000'
    test_url = addr + '/pred2'

    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    # img = cv2.imread('../img-tmp/black-knight-2.jpg')

    _, img_encoded = cv2.imencode('.jpg', img)

    response = requests.post(test_url, 
                            data=img_encoded.tostring(), 
                            headers=headers)
    response_json = response.json()
    q.put(response_json['pred-class-name'])
    print(response_json)

def draw_rect(img, rect, color='yellow', thick=3):
    COLOR = (0, 255, 255)
    if color == 'blue':
        COLOR = (255,0,0)
    cv2.rectangle(img, rect[0], rect[1], COLOR, thick)
    return img

def draw(frame):
    '''450, 600 '''
    frame = draw_rect( frame
                    ,((200,50), (400, 400))
                    ,color='yellow'
                    ,thick = 3
                    )
    return frame

def annotate(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(frame,text,(50,50), font, 1.4,
                        (255,255,255),2,cv2.LINE_AA)

def annotate_class(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(frame,text,(50,40), font, 1.4,
                        (255,255,255),2,cv2.LINE_AA)

def annotate_think(frame, b_snap, think_counter):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if think_counter < 1:
        text =  'SNAP '
    else:
        text =  'proc '

    if think_counter > 0:
        text += (think_counter * '.')

    return cv2.putText(frame,text,(50,440), font, 1.2,
                        (255,255,255),2,cv2.LINE_AA)
    

# setup
cam_num = 1
camera = cv2.VideoCapture(cam_num)
counter = 0
N = 60

q = queue.Queue()
b_starting_thread = True
response_text = None

b_reset_think = True
b_snap = False
think_counter = 0
mod_think_counter = 3


#loop -----------------------
while(camera.isOpened()):

    grabbed, frame = camera.read()
    frame = imutils.resize(frame, width=600)
    frame = draw(frame)
    
    try:
        response_text = q.get(False)  
        b_reset_think = True
    except queue.Empty:
        pass
    except:
        response_text = "idk? exception in get"

    if (response_text is None) and b_starting_thread:
        response_text = 'starting...'
        b_starting_thread = False
        
    frame = annotate_class(frame, response_text)
    
    if not(args['thinkoff']):

        frame = annotate_think( frame,
                                b_snap, 
                                int(think_counter // mod_think_counter))
    
    cv2.imshow("Hello Video", frame)

    if counter % N == 0:

        print('snap')
        b_snap = True
        think_counter = 0
        t0 = time.time()
        
        t = threading.Thread(target = ping, 
                             args = (frame[50:400,200:400,:],q) )
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



