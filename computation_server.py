from flask import Flask, request
import numpy as np
import json
import time

app = Flask(__name__)
@app.route('/mlp', methods = ['POST'])
def MLP():
    transfer_begin = time.time()
    print('RRRRRRR: ' + str(time.time()))
    x = request.form.get('x')
    w = request.form.get('w')

    x = np.asarray(json.loads(x))
    w = np.asarray(json.loads(w))
    begin_time = time.time()
    ans = np.matmul(x, w)
    end_time = time.time()
    cost_time = end_time - begin_time
    print(cost_time)
    ans = json.dumps(ans.tolist())
    transfer_end = time.time()
    print('Transfer Cost: ' + str(transfer_end-transfer_begin))
    return ans

if __name__ == '__main__':
    app.run()