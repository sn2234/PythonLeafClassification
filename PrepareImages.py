import SetEnvForGpu

import os
import numpy as np
np.random.seed(123456)  # for reproducibility

from os.path import join

import sklearn.metrics as metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt
from scipy.misc import imsave

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D
from keras.utils import np_utils

import DataModel
import Model

# Define some helper functions

def loadImages(ids):
    imgPaths = [join(DataModel.trainImagesDir, "{0}.jpg".format(path)) for path in ids]
    imgs = [image.load_img(path, grayscale=True, target_size=(Model.img_dim_x, Model.img_dim_y))
            for path in imgPaths
            ]
    imgArrays = [image.img_to_array(i) for i in imgs]
    
    return imgArrays

# Load data in approptiate format    
csvSubset = DataModel.imageSpecies #.sample(900)
subsetSize = csvSubset.shape[0]
imgArrays = loadImages(csvSubset.id)

targetPath = os.path.join(DataModel.trainImagesDir, "Transformed")

if not os.path.exists(targetPath):
    os.makedirs(targetPath)

for i in range(len(imgArrays)):
    img = imgArrays[i].reshape(imgArrays[i].shape[1], imgArrays[i].shape[2])
    imsave(os.path.join(targetPath, "{0}.jpg".format(csvSubset.id[i])), img)
