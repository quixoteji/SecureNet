import time
import requests
import json
import pickle
import numpy as np

def cloud_cal(X, W):
    time1 = time.time()
    x = json.dumps(X.tolist())
    w = json.dumps(W.tolist())
    data = {'x': x, 'w': w}
    time2 = time.time()
    print("Package time: " + str(time2 - time1))
    ans = requests.post("http://127.0.0.1:5000/mlp_json", data=data)
    time3 = time.time()
    ans = np.asarray(json.loads(ans.text))
    print("Transmission time: " + str(time3-time2))
    return ans
    # return cost_time

def cloud_pickle_cal(X, W):
    time1 = time.time()
    x = pickle.dumps(X)
    w = pickle.dumps(W)
    data = pickle.dumps((X,W))
    time2 = time.time()
    print("Package time: " + str(time2 - time1))
    ans = requests.post("http://127.0.0.1:5000/mlp_pickle", data=data)
    ans = pickle.loads(ans.content)
    time3 = time.time()
    print("Transmission time: " + str(time3-time2))
    return ans
if __name__ == '__main__':
    x = np.random.rand(1000,1000)
    w = np.random.rand(1000,1000)

    a = cloud_cal(x, w)
    b = cloud_pickle_cal(x, w)
    if((a==b).all()):
        print('OK! ')
    else:
        print('Not OK!')


