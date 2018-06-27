from examples import loaderANN

model = loaderANN()
# model summary
'''
________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 512)               401920
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
'''
# model.summary()
layer1 = model.layers[0]
layer2 = model.layers[2]
layer3 = model.layers[4]

# model layer get_weights
w1 = layer1.get_weights()[0]
b1 = layer1.get_weights()[1]

w2 = layer2.get_weights()[0]
b2 = layer2.get_weights()[1]

w3 = layer3.get_weights()[0]
b3 = layer3.get_weights()[1]

import numpy as np
np.save('w1', w1)
np.save('b1', b1)
np.save('w2', w2)
np.save('b2', b2)
np.save('w3', w3)
np.save('b3', b3)
