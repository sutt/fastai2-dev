import time
import requests
import json
import sys

b_mini = False
if len(sys.argv) > 1:
    b_mini = True

url = 'http://127.0.0.1:5000/hello'

if b_mini: url = 'http://127.0.0.1:5000/'

print('sending...')

if b_mini:
    t0 = time.time()
    r = requests.get( url)
    print(round(time.time() - t0, 2))
else:
    r = requests.post( url, json={"time": str(time.time()) } )

print('done')
