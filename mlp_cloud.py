import requests
import json

import keras
import numpy as np
import time
import pickle

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
    (rows, cols) = X.shape
    res1 = np.zeros_like(X)
    for i in range(rows):
        res1[i] = MA[i][i] * X[i]
    # print('#####')
    # print(res1)
    res2 = np.zeros_like(X)
    for j in range(cols):
        res2[:,j] = MB[j][j] * res1[:,j]
    return res2



def denoise(MA, X, MB):
    '''
    First is the old version of noise and denoise
    Now replaced by the new version of noise and denoise
    '''
    # medium = np.matmul(MA, X)
    # ans = np.matmul(medium, MB)
    return noise(MA, X, MB)

def cloud_cal(X, W):
    '''
    Json version
    '''
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

def cloud_pickle_cal(X, W):
    '''
    Pickle version
    '''
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
        batch_x = x[i*batch_size : (i+1)*batch_size]
        batch_y = y[i*batch_size : (i+1)*batch_size]

        batch_x = input_convert(batch_x, batch_size)
        nx = noise(M0, batch_x, M1)
        nW1 = noise(inv_M1, W1, M2)

        l1_from_cloud = cloud_pickle_cal(nx, nW1)

        l1 = denoise(inv_M0, l1_from_cloud, inv_M2)
        l1 = hidden_layer(l1)

        l1 = input_convert(l1, batch_size)
        nl1 = noise(M0, l1, M3)
        nW2 = noise(inv_M3, W2, M2)

        l2_from_cloud  = cloud_pickle_cal(nl1, nW2)

        l2 = denoise(inv_M0, l2_from_cloud, inv_M2)
        l2 = hidden_layer(l2)

        l2 = input_convert(l2, batch_size)
        nl2 = noise(M0, l2, M3)
        nW3 = noise(inv_M3, W3, M4)

        l3_from_cloud  = cloud_pickle_cal(nl2, nW3)

        l3 = denoise(inv_M0, l3_from_cloud, inv_M4)

        for j in range(batch_size):
            out[j] = softmax(l3[j])
            out[j] = np.argmax(out[j])

        break

    end = time.time()
    cost = end-begin
    print('done')
    return cost

if __name__ == '__main__':

    # # Test for noise and denoise
    # size = 10000
    # a = np.random.rand(size, size)
    # ma = np.random.rand(size, size) * np.eye(size)
    # mb = np.random.rand(size, size) * np.eye(size)
    # time1 = time.time()
    # A = noise(ma, a, mb)
    # time2 = time.time()
    # B = denoise(ma, a, mb)
    # time3 = time.time()
    # # print(A)
    # # print(B)
    # print('NOISE TIME: ' + str(time2 - time1))
    # print('NOISE TIME: ' + str(time3 - time2))
    # print(np.equal(A,B).all())

    # Time cost for whole system
    cost = []
    for i in range(1000, 3000, 500):
        cost_time = mlp_cloud(i)
        cost.append(cost_time)
    print(cost)
    #[0.37601780891418457, 0.4062159061431885, 0.5541269779205322, 0.6482551097869873]
    #[0.05959296226501465, 0.09139704704284668, 0.12113618850708008, 0.14902901649475098]
    
    

