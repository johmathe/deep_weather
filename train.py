import numpy as np

from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import MaxPooling2D, Convolution2D
from keras.layers.core import Flatten, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD

import h5py
np.random.seed(1337)  # for reproducibility

print 'loading data...'
dest = './weather.hdf5'  # was '/home/johmathe/weather.hdf5'
data = h5py.File(dest)

X_train = data['X_train'][...]
Y_train = data['Y_train'][...]
X_test = data['X_val'][...]
Y_test = data['Y_val'][...]

print ('x shape:', X_train.shape)
print ('y shape:', Y_train.shape)
# /!\ with these shapes new version of Kerias needs
# "image_dim_ordering": "th" in ~/.keras/keras.json

print 'normalizing data...'
mean = np.mean(X_train)
std = 3*np.std(X_train)
print 'mean: %f' % mean
print 'std: %f' % std
X_train -= mean
X_train /= std

Y_train -= mean
Y_train /= std

X_test -= mean
X_test /= std

Y_test -= mean
Y_test /= std


init_method = 'he_normal'

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(1, 361, 720), init=init_method))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, init=init_method))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, init=init_method))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, init=init_method))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, init=init_method))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dense(128, init=init_method))
model.add(PReLU())
model.add(Dense(1, init=init_method))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.95, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=16, nb_epoch=5, shuffle=True)
score = model.evaluate(X_test, Y_test)
print score
