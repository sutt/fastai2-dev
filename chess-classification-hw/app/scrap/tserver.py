import sys, time
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def base():
    return 'ok'

@app.route('/hello',methods=['GET', 'POST'] )
def hello():
    r = request
    t = time.time()
    elapsed = t - float(r.json['time'])
    print(round(elapsed,5))
    sys.stdout.flush()
    return ''
    

if __name__ == "__main__":
    app.run(debug=False, threaded=True)