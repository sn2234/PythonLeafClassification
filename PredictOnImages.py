
import numpy as np
np.random.seed(123456)  # for reproducibility

from os.path import join

import sklearn.metrics as metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import DataModel

# Network params
batch_size = 128
nb_classes = 99
nb_epoch = 12

img_dim_x=200
img_dim_y=200
input_shape = (1,img_dim_y,img_dim_x)

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

csvSubset = DataModel.imageSpecies.sample(10)

imgPaths = [join(DataModel.trainImagesDir, "{0}.jpg".format(path)) for path in csvSubset.id]
imgs = [image.load_img(path, grayscale=True, target_size=(img_dim_x, img_dim_y))
        for path in imgPaths
        ]
imgArrays = [image.img_to_array(i) for i in imgs]

y = csvSubset.species_id.values
y_cat = np_utils.to_categorical(y, nb_classes)
x = np.vstack(imgArrays).reshape((x.shape[0], 1, img_dim_y, img_dim_x))