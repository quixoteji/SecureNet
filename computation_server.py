from flask import Flask, request, Response 
import numpy as np
import pickle
import json
import time

app = Flask(__name__)
@app.route('/mlp_json', methods = ['POST'])
def MLP_json():
    x = request.form.get('x')
    w = request.form.get('w')
    x = np.asarray(json.loads(x))
    w = np.asarray(json.loads(w))
    ans = np.matmul(x, w)
    ans = json.dumps(ans.tolist())
    return ans

@app.route('/mlp_pickle', methods = ['POST'])
def mlp_pickle():
    x = request.get_data()
    x = pickle.loads(x)
    (a,b) = x
    res = np.matmul(a, b)
    ans = pickle.dumps(res)
    return Response(ans, status = 200)

if __name__ == '__main__':
    app.run()