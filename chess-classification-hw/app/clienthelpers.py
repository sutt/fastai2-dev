import threading
import queue
import requests
import json
import cv2
import imutils
import time
from matplotlib import pyplot as plt

x_axis = [
    'black-bishop',
    'black-king',
    'black-knight',
    'black-pawn',
    'black-queen',
    'black-rook',
    'white-bishop',
    'white-king',
    'white-knight',
    'white-pawn',
    'white-queen',
    'white-rook']

def chart_it(y, x_axis=x_axis):
    plt.bar(x_axis, y)
    plt.show()

def ping(img, q, b_debug=False):
    
    addr = 'http://localhost:5000'
    test_url = addr + '/pred'

    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    # img = cv2.imread('../img-tmp/black-knight-2.jpg')

    _, img_encoded = cv2.imencode('.jpg', img)

    response = requests.post(test_url, 
                            data=img_encoded.tostring(), 
                            headers=headers)
    response_json = response.json()
    q.put(response_json['pred-class-name'])
    if b_debug: print(response_json)

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
    