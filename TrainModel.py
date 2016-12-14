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
import Model
import PrepareImages

# Load data in approptiate format    
csvSubset = DataModel.imageSpecies #.sample(900)
subsetSize = csvSubset.shape[0]
imgArrays = PrepareImages.loadImages(DataModel.trainImagesDir, csvSubset.id, Model.img_dim_x)
y = csvSubset.species_id.values
x = np.vstack(imgArrays).reshape((len(imgArrays), 1, Model.img_dim_y, Model.img_dim_x))
(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(x, y, test_size=0.2)
y_train_cat = np_utils.to_categorical(y_train, Model.nb_classes)
y_cv_cat = np_utils.to_categorical(y_cv, Model.nb_classes)

model = Model.createModelDefinition()

def fitWithGenerator():
    # Use generator to produce additional images
    datagen = image.ImageDataGenerator(rotation_range=360,
                        #shear_range=0.2,
                        #zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest')

    #model.fit_generator(datagen.flow(x_train, y_train_cat), samples_per_epoch = len(x_train), nb_epoch = 20)
    epoch = 0
    maxEpochs = 10
    itemsPerBatch = 200
    trainBatchSize = 25
    for X_batch, Y_batch in datagen.flow(x_train, y_train_cat, batch_size=itemsPerBatch):

        iter = 0
        iFrom = 0
        while iFrom < X_batch.shape[0]:
            print("Within epoch batch: [{0}, {1}]".format(iFrom, iFrom+trainBatchSize))
            X_current = X_batch[iFrom:(iFrom+trainBatchSize)]
            Y_current = Y_batch[iFrom:(iFrom+trainBatchSize)]
            loss = model.train_on_batch(X_current, Y_current)
            print("Epoch: {0}.{1}, Loss: {2}".format(epoch, iter, loss))

            iFrom = iFrom + trainBatchSize
            iter = iter + 1

        epoch = epoch + 1
        if epoch > maxEpochs:
            break


def fitWithRawData():
    model.fit(x_train, y_train_cat, nb_epoch=30)

def eval():
    return model.evaluate(x_cv, y_cv_cat)

def saveTrainedWeights(fileName):
    model.save_weights(fileName)

