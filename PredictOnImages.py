
import numpy as np
np.random.seed(123456)  # for reproducibility

from os.path import join

import sklearn.metrics as metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D
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

def loadImages(ids):
    imgPaths = [join(DataModel.trainImagesDir, "{0}.jpg".format(path)) for path in ids]
    imgs = [image.load_img(path, grayscale=True, target_size=(img_dim_x, img_dim_y))
            for path in imgPaths
            ]
    imgArrays = [image.img_to_array(i) for i in imgs]
    
    return imgArrays
    
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.1))

model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1],
                        border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.1))

model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1],
                        border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1],
                        border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

csvSubset = DataModel.imageSpecies.sample(900)

imgArrays = loadImages(csvSubset.id)

y = csvSubset.species_id.values
x = np.vstack(imgArrays).reshape((len(imgArrays), 1, img_dim_y, img_dim_x))

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(x, y, test_size=0.2)


y_train_cat = np_utils.to_categorical(y_train, nb_classes)

# Use generator to produce additional images
datagen = image.ImageDataGenerator(
                    rotation_range=90,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

model.fit_generator(datagen.flow(x_train, y_train_cat), samples_per_epoch = 100, nb_epoch = 30)

model.fit(x_train, y_train_cat, nb_epoch=20)
#model.save_weights('model1.bin')
#model.load_weights('model1.bin')

y_cv_cat = np_utils.to_categorical(y_cv, nb_classes)

model.evaluate(x_cv, y_cv_cat)

#imgArraysTest = loadImages(DataModel.csvTest.id)
#x_test = np.vstack(imgArraysTest).reshape((len(imgArraysTest), 1, img_dim_y, img_dim_x))
#y_test_cat = model.predict(x_test)
#of = DataModel.prepareOutput(y_test_cat, DataModel.csvTest.id.values)

#of.to_csv('out.csv', index=False)
