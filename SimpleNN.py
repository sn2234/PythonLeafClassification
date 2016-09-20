import numpy as np
import sklearn.metrics as metrics
from sklearn import cross_validation

import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils

import DataModel

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(DataModel.x, DataModel.y, test_size=0.2)

print("Initializing model...")

nb_classes = 99

model = Sequential()
model.add(Dense(2000, input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
print("Fitting model...")

y_train_cat = np_utils.to_categorical(y_train, nb_classes)
y_cv_cat = np_utils.to_categorical(y_cv, nb_classes)

model.fit(x_train, y_train_cat)

y_pred = model.predict(x_cv)

score, acc = model.evaluate(x_cv, y_cv_cat)
print("Accuracy on CV dataset: {0}".format(acc))
