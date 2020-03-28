'''
https://stackoverflow.com/questions/2846653/how-can-i-use-threading-in-python

'''

import queue
import threading
import requests

# Called by each thread
def get_url(q, url):
    q.put(requests.get(url).content)

theurls = ["http://google.com", "http://yahoo.com", 
            "http://127.0.0.1:5000/"]

q = queue.Queue()

for u in theurls:
    t = threading.Thread(target=get_url, args = (q,u))
    t.daemon = True
    t.start()

s = q.get()
print( s)
q.get()
