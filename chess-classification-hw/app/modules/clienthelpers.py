import threading
import queue
import requests
import json
import cv2
import imutils
import time
from matplotlib import pyplot as plt


def ping(img, q, b_debug=False):
    
    addr = 'http://localhost:5000'
    test_url = addr + '/pred'

    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    _, img_encoded = cv2.imencode('.jpg', img)

    if b_debug: print('~~~~ sending request...\n')
    
    try:
        response = requests.post(test_url, 
                                data=img_encoded.tostring(), 
                                headers=headers)
        
        response_json = response.json()
        
        q.put(response_json['pred-class-name'])
        
        if b_debug: print(response_json)
    
    except Exception as e:
        print('Could not connect to server; make sure predserver.py is running')
        # print(e)  #uncomment for debugging
    
    if b_debug: print('\n~~~~ end of request...')


def draw_rect(img, rect, color='yellow', thick=3):
    COLOR = (0, 255, 255)
    if color == 'blue':
        COLOR = (255,0,0)
    cv2.rectangle(img, rect[0], rect[1], COLOR, thick)
    return img


def draw(frame, rect, thick=3):
    '''450, 600 '''
    # ((x0,  y0),  (x1, y1))
    x0, y0, x1, y1 = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
    frame = draw_rect( frame
                    ,((x0-thick, y0-thick),(x1+thick, y1+thick))
                    ,color='yellow'
                    ,thick = thick
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
    