import numpy as np
import sklearn.metrics as metrics
from sklearn import cross_validation

import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import Sequential

import DataModel

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(DataModel.x, DataModel.y, test_size=0.2)

