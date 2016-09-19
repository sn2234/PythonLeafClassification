
import numpy as np
import sklearn.metrics as metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import DataModel

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(DataModel.x, DataModel.y, test_size=0.2)


rfClf = RandomForestClassifier()
rfClf.fit(x_train, y_train)
y_pred = rfClf.predict(x_cv)

metrics.accuracy_score(y_cv, y_pred)
metrics.confusion_matrix(y_cv, y_pred)
#metrics.classification_report(y_cv, y_pred)

