import requests
import json

import keras
import numpy as np
import time

from mlp_local import weights_load
from mlp_local import load_data
from mlp_local import hidden_layer
from mlp_local import input_convert
from mlp_local import softmax
'''
In this verion, I use the numpy to conduct matrix multiplication and flask jsonify to conduct data jsonified
'''

'''
input (n, 785)  W1 (785, 512)   l1 (n, 512)
input (n, 513)  W2 (513, 512)   l2 (n, 512)
input (n, 513)  W3 (513, 10)    l3 (n, 10)
-------------------------------------------
nInput (n, n)   nInput-W1 (785, 785) W1 (512, 512)
nInput (n, n)   nInput-W2 (513, 513) W2 (512, 512)
nInput (n, n)   nInput-W3 (513, 513) W3 (10, 10)
-------------------------------------------
noise-matrix
M0 (n,n) inv_M0
M1 (785, 785) inv_M1
M2 (512, 512) inv_M2
M3 (513, 513) inv_M3
M4 (10, 10)   inv_M4
'''

def M_generator(x):
    seed = np.random.rand(x, x)
    inv_seed = 1/seed
    M = seed * np.eye(x)
    inv_M = inv_seed * np.eye(x)
    return M, inv_M

def cloud_layer_cal(x, W, batch_size):
    x = input_convert(x, batch_size)
    out = np.matmul(x, W)
    return out

def noise(MA, X, MB):
    medium = np.matmul(MA, X)
    ans = np.matmul(medium, MB)
    return ans

def denoise(MA, X, MB):
    medium = np.matmul(MA, X)
    ans = np.matmul(medium, MB)
    return ans

def cloud_cal(X, W):
    print('Package Begine:')
    begin = time.time()
    x = json.dumps(X.tolist())
    w = json.dumps(W.tolist())
    data = {'x': x, 'w': w}
    print('Packag Time: ' + str(time.time()-begin))
    print('SSSSSS: ' + str(time.time()))
    ans = requests.post("http://127.0.0.1:5000/mlp", data=data)
    end = time.time()
    cost_time = end-begin
    return np.asarray(json.loads(ans.text)), cost_time

def mlp_cloud(batch_size):
    x, y = load_data()
    M0, inv_M0 = M_generator(batch_size)
    M1, inv_M1 = M_generator(785)
    M2, inv_M2 = M_generator(512)
    M3, inv_M3 = M_generator(513)
    M4, inv_M4 = M_generator(10)
    W1, W2, W3 = weights_load()
    print('begin')
    local_time = 0
    out = [0 for i in range(batch_size)]
    begin = time.time()
    for i in range(round(len(y)/batch_size)):   
        print('enter')
        batch_x = x[i*batch_size : (i+1)*batch_size]
        batch_y = y[i*batch_size : (i+1)*batch_size]

        batch_x = input_convert(batch_x, batch_size)
        nx = noise(M0, batch_x, M1)
        nW1 = noise(inv_M1, W1, M2)
        print('send1')
        l1_from_cloud, cost_time1 = cloud_cal(nx, nW1)
        print('receive1')
        l1 = denoise(inv_M0, l1_from_cloud, inv_M2)
        l1 = hidden_layer(l1)

        l1 = input_convert(l1, batch_size)
        nl1 = noise(M0, l1, M3)
        nW2 = noise(inv_M3, W2, M2)
        print('send2')
        l2_from_cloud, cost_time2 = cloud_cal(nl1, nW2)
        print('receive2')
        l2 = denoise(inv_M0, l2_from_cloud, inv_M2)
        l2 = hidden_layer(l2)

        l2 = input_convert(l2, batch_size)
        nl2 = noise(M0, l2, M3)
        nW3 = noise(inv_M3, W3, M4)
        print('send3')
        l3_from_cloud, cost_time3 = cloud_cal(nl2, nW3)
        print('receive3')
        l3 = denoise(inv_M0, l3_from_cloud, inv_M4)

        for j in range(batch_size):
            out[j] = softmax(l3[j])
            out[j] = np.argmax(out[j])

        break

    end = time.time()
    cost = end-begin-cost_time1-cost_time2-cost_time3
    print('done')
    return cost

if __name__ == '__main__':
    # y, out = mlp_cloud(10000)
    time1 = mlp_cloud(500)
    
    

