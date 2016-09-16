
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

trainDescFile = "..\\Data\\train.csv"
trainImagesDir = "..\\Data\\images"

csvRaw = pd.read_csv(trainDescFile)

enc = preprocessing.LabelEncoder()

csvRaw['species_id'] = enc.fit_transform(csvRaw.species).astype(int)

excludedFromTrain = {'id', 'species', 'species_id'}

x = csvRaw[[c for c in csvRaw.columns if c not in excludedFromTrain]].values
y = csvRaw.species_id.values

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(x, y, test_size=0.2)


rfClf = RandomForestClassifier()
rfClf.fit(x_train, y_train)
y_pred = rfClf.predict(x_cv)

metrics.accuracy_score(y_cv, y_pred)
metrics.confusion_matrix(y_cv, y_pred)
#metrics.classification_report(y_cv, y_pred)

