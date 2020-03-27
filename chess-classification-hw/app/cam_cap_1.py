import cv2
import imutils

cam_num = 1

camera = cv2.VideoCapture(cam_num)

# while True:
while(camera.isOpened()):

    (grabbed, frame) = camera.read()
    
    # print(grabbed)
    # print(type(frame))
    
    frame = imutils.resize(frame, width=600)

    cv2.imshow("Hello Video", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
