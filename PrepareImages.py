import math
import os
import numpy as np
np.random.seed(123456)  # for reproducibility

from os.path import join

import matplotlib.pyplot as plt
from scipy.misc import imsave, imread, imresize


# Define some helper functions
def rescaleToSquare(img, squareSide):
    (imgHeight, imgWidth) = img.shape
    
    # Calculate scale factor
    scaleFactor = squareSide/max(imgHeight, imgWidth)

    rescaledImage = imresize(img, scaleFactor)
    (newHeight, newWidth) = rescaledImage.shape
    
    #assert(newHeight == squareSide or newWidth == squareSide)
    
    # Center and pad rescaled image with zeros
    if newHeight < squareSide:
        padLen = squareSide - newHeight
        padTopLen = padLen//2
        padBottomLen = padLen//2 + padLen%2
        padTop = np.zeros((padTopLen, newWidth))
        padBottom = np.zeros((padBottomLen, newWidth))
        rescaledImage = np.vstack((padTop, rescaledImage, padBottom))

    if newWidth < squareSide:
        padLen = squareSide - newWidth
        padLeftLen = padLen//2
        padRightLen = padLen//2 + padLen%2
        padLeft = np.zeros((squareSide, padLeftLen))
        padRight = np.zeros((squareSide, padRightLen))
        rescaledImage = np.hstack((padLeft, rescaledImage, padRight))

    return rescaledImage

def processImage(imgPath, squareSide):
    rawImg = imread(imgPath, mode = 'L')
    return rescaleToSquare(rawImg, squareSide)
    
def loadImages(imagesDir, ids, squareSide):
    imgPaths = [join(imagesDir, "{0}.jpg".format(path)) for path in ids]
    imgs = [processImage(path, squareSide)
            for path in imgPaths
            ]
    
    return imgs
