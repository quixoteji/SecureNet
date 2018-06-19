import keras
from keras.models import model_from_json, load_model
import json
import os
import sys

# models path
PATH = os.path.abspath('./models/')
# weights path
WPATH = os.path.join(PATH, 'mnist_mlp_weights.h5')
# architecture path
APATH = os.path.join(PATH, 'mnist_mlp.json')

def loaderANN():
    '''
    Load model ANN
    '''
    with open(APATH, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(WPATH)
    return model 

def loaderCNN():
    '''
    Load model CNN
    '''
    with open('./models/mnist_cnn.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('./models/mnist_cnn_weights.h5')
    return model