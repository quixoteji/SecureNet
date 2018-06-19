import time
import requests
import json
import numpy as np

def cloud_cal(X, W):
    begin = time.time()
    x = json.dumps(X.tolist())
    w = json.dumps(W.tolist())
    data = {'x': x, 'w': w}
    ans = requests.post("http://127.0.0.1:5000/mlp", data=data)
    end = time.time()
    cost_time = end-begin
    return np.asarray(json.loads(ans.text)), cost_time

if __name__ == '__main__':
    x = np.random.rand(4,4)
    w = np.random.rand(4,4)
    a, cost_time = cloud_cal(x, w)
    print(a)
    print(cost_time)
    # print(timeA)
