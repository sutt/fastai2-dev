import requests
import json
import cv2

addr = 'http://localhost:5000'
# test_url = addr + '/api/test'
test_url = addr + '/pred'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('../img-tmp/black-knight-2.jpg')
print(img.shape)

_, img_encoded = cv2.imencode('.jpg', img)

response = requests.post(test_url, 
                        data=img_encoded.tostring(), 
                        headers=headers)

# print(json.loads(response.text))
print(response.text)
