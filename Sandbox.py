import math
import os
import numpy as np
np.random.seed(123456)  # for reproducibility

from os.path import join

import matplotlib.pyplot as plt
from scipy.misc import imsave, imread, imresize

from keras.preprocessing import image

import DataModel
import Model
import PrepareImages

def transformImages():

    csvSubset = DataModel.imageSpecies #.sample(900)
    subsetSize = csvSubset.shape[0]
    assert(Model.img_dim_x == Model.img_dim_y)
    imgArrays = PrepareImages.loadImages(DataModel.trainImagesDir, csvSubset.id, Model.img_dim_x)

    targetPath = join(DataModel.trainImagesDir, "Transformed")

    if not os.path.exists(targetPath):
        os.makedirs(targetPath)

    for i in range(len(imgArrays)):
        img = imgArrays[i]
        imsave(join(targetPath, "{0}.jpg".format(csvSubset.id[i])), img)


def transformWithGenerator():
    csvSubset = DataModel.imageSpecies #.sample(900)
    subsetSize = csvSubset.shape[0]
    assert(Model.img_dim_x == Model.img_dim_y)
    imgArrays = PrepareImages.loadImages(DataModel.trainImagesDir, csvSubset.id, Model.img_dim_x)

    targetPath = join(DataModel.trainImagesDir, "Generator")

    if not os.path.exists(targetPath):
        os.makedirs(targetPath)

    datagen = image.ImageDataGenerator(
                    rotation_range=360,
                    #shear_range=0.2,
                    #zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

    y = csvSubset.species_id.values
    x = np.array(imgArrays).reshape((len(imgArrays), 1, Model.img_dim_y, Model.img_dim_x))

    epoch = 0
    itemsPerBatch = 512
    for X_batch, Y_batch in datagen.flow(x, y, batch_size=itemsPerBatch):

        for i in range(Y_batch.shape[0]):
            fileName = "epoch_{1}_class_{0}_id{2}.jpg".format(Y_batch[i], epoch, i)
            img = X_batch[i,:,:,:].reshape((Model.img_dim_y, Model.img_dim_x))
            imsave(join(targetPath, fileName), img)

        epoch = epoch + 1

        if epoch > 5:
            break
    
##testImage = join(DataModel.trainImagesDir, "1065.jpg")
#testImage = join(DataModel.trainImagesDir, "10.jpg")

#rawImg = imread(testImage, mode = 'L')

