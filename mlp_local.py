import keras
import numpy as np
import time

def weights_load():
    '''
    W1 (785, 512)
    W2 (513, 512)
    W3 (513, 10)
    '''
    w1 = np.load('./weightsMLP/w1.npy')
    b1 = np.load('./weightsMLP/b1.npy').reshape(1,512)
    # b1 = np.tile(b1, (num_samples, 1))
    w2 = np.load('./weightsMLP/w2.npy')
    b2 = np.load('./weightsMLP/b2.npy').reshape(1,512)
    # b2 = np.tile(b2, (num_samples, 1))
    w3 = np.load('./weightsMLP/w3.npy')
    b3 = np.load('./weightsMLP/b3.npy').reshape(1, 10)
    # b3 = np.tile(b3, (num_samples, 1))

    W1 = np.concatenate((w1, b1), axis = 0)
    W2 = np.concatenate((w2, b2), axis = 0)
    W3 = np.concatenate((w3, b3), axis = 0)
    return W1, W2, W3

def input_convert(x, batch_size):
    padding = np.ones([batch_size, 1])
    x = np.concatenate((x,padding), axis = 1)
    return x

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x = np.concatenate((x_train, x_test), axis = 0).reshape(70000, 28*28).astype('float32')/255
    y = np.concatenate((y_train, y_test), axis = 0)
    return x, y

def layer_cal(x, w, num_samples):
    x = input_convert(x, num_samples)
    out = np.matmul(x, w)
    return out

def hidden_layer(out):
    return np.maximum(out, 0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def mlp_local(num_samples):
    '''
    local run mlp by numpy
    return:
    y: label
    out: output
    cost: runtime
    '''
    x, y = load_data()
    x = x[0:num_samples]
    y = y[0:num_samples]
    W1, W2, W3 = weights_load()
    out = [0 for i in range(num_samples)]

    begin = time.time()
    # for 
    l1 = layer_cal(x, W1, num_samples)
    l1 = hidden_layer(l1)
    l2 = layer_cal(l1, W2, num_samples)
    l2 = hidden_layer(l2)
    l3 = layer_cal(l2, W3, num_samples)

    for i in range(num_samples):
        out[i] = softmax(l3[i])
        out[i] = np.argmax(out[i])
    end = time.time()

    cost = end - begin
    return cost

if __name__ == '__main__':
    y, out, cost = mlp_local(10000)

        
