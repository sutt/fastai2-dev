import requests
import json
import cv2
import imutils


def ping(img):
    
    addr = 'http://localhost:5000'
    test_url = addr + '/pred'

    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    # img = cv2.imread('../img-tmp/black-knight-2.jpg')

    _, img_encoded = cv2.imencode('.jpg', img)

    response = requests.post(test_url, 
                            data=img_encoded.tostring(), 
                            headers=headers)
    print(response.text)

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

# setup
cam_num = 1
camera = cv2.VideoCapture(cam_num)
counter = 0
N = 60

#loop -----------------------
while(camera.isOpened()):

    grabbed, frame = camera.read()
    
    frame = imutils.resize(frame, width=600)

    frame = draw(frame)
    cv2.imshow("Hello Video", frame)
    
    if counter % N == 0:
        print('pinging...')
        print(frame.shape)
        ping(frame[50:400,200:400,:])
    counter += 1

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()



