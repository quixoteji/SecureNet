import numpy as np
import re
import base64
from scipy.misc import imsave, imread, imresize
from examples import loaderANN

from flask import Flask, render_template, request, url_for
# initialize Flask 
global model
model = loaderANN()

app = Flask(__name__)
# model = loaderANN()
def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))

def noiseImage(img):
    seed = np.random.rand(28,28)
    noiseA = np.eye(28) * seed
    noiseB = np.eye(28) * (1/seed)
    print(noiseA)
    print(noiseB)
    print(np.matmul(noiseA, noiseB))
    noised = np.matmul(np.matmul(noiseA, img), noiseB)
    # noised = np.invert(noised)
    noised = imresize(noised, (280,280))
    imsave('static/noise.png', noised)
    
@app.route('/')
def index():
    print("ENTER INDEX")
    if not request.script_root:
        request.script_root = url_for('.index', _external=True)
    return render_template('index.html')

@app.route('/predict/',methods=['POST'])
def predict():
    # imgData = request.get_data()
    # convertImage(imgData)
    print("ENTER PREDICTION")
    imgData = request.get_data()
    convertImage(imgData)
    print(type(imgData))
    x = imread('output.png',mode='L')
    print(x.shape)
    x = np.invert(x)
    x = imresize(x,(28,28))
    noiseImage(x)
    # print(x)
    x = x.reshape(1, 784)
    y = model.predict(x)
    y = np.argmax(y)
    return str(y);

if __name__ == "__main__":
    # print(("* Loading Keras model and Flask starting server..."
	# 	"please wait until server has fully started"))
    # port = int(os.environ.get('PORT', 8000))
    app.run()
