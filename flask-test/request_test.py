from flask import Flask, request, jsonify
import numpy as np
import json

app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello world'

@app.route('/test', methods=['POST'])
def test():
    # print(request.headers)
    # print(request.form)
    # print(request.form['name'])
    # print(request.form.get('name'))
    # print(request.form.getlist('name'))
    # print(request.form.get('nickname', default='little apple'))
    print(request.form)
    a = request.form.get('a')
    b = request.form.get('b')

    a = np.asarray(json.loads(a))
    b = np.asarray(json.loads(b))
    
    c = np.matmul(a,b)
    c = c.tolist()
    c = json.dumps(c)
    return c

if __name__ == '__main__':
    app.run()