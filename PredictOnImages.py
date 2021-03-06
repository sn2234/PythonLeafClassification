
import SetEnvForGpu

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
import PrepareImages

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
model.add(Dropout(0.3))

model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1],
                        border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.3))

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
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

csvSubset = DataModel.imageSpecies #.sample(900)

subsetSize = csvSubset.shape[0]

imgArrays = PrepareImages.loadImages(DataModel.trainImagesDir, csvSubset.id, img_dim_x)

y = csvSubset.species_id.values
x = np.vstack(imgArrays).reshape((len(imgArrays), 1, img_dim_y, img_dim_x))

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(x, y, test_size=0.2)


y_train_cat = np_utils.to_categorical(y_train, nb_classes)

# Use generator to produce additional images
datagen = image.ImageDataGenerator(
                    rotation_range=90,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    #rescale=1./255,
                    shear_range=0.2,
                    #zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

model.fit_generator(datagen.flow(x_train, y_train_cat), samples_per_epoch = len(x_train), nb_epoch = 20)

model.fit(x_train, y_train_cat, nb_epoch=100)

#model.save_weights('model1.bin')
#model.load_weights('model1.bin')

y_cv_cat = np_utils.to_categorical(y_cv, nb_classes)

model.evaluate(x_cv, y_cv_cat)

#imgArraysTest = loadImages(DataModel.csvTest.id)
#x_test = np.vstack(imgArraysTest).reshape((len(imgArraysTest), 1, img_dim_y, img_dim_x))
#y_test_cat = model.predict(x_test)
#of = DataModel.prepareOutput(y_test_cat, DataModel.csvTest.id.values)

#of.to_csv('out.csv', index=False)

nb_epoch=1
for e in range(nb_epoch):
    print('Epoch: {0}'.format(e))
    batches = 0
    for X_batch, Y_batch in datagen.flow(x_train, y_train_cat, batch_size=32):
        print("Training on batch: X_batch: {0}, Y_batch: {1}".format(X_batch.shape, Y_batch.shape))
        print("Sample image, class: {0}".format(Y_batch[0]))
        plt.imshow(image.array_to_img(X_batch[0]))
        loss = model.train_on_batch(X_batch, Y_batch)
        print("Loss: {0}".format(loss))
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

#xx = datagen.flow(x_train, y_train_cat, batch_size=50)
