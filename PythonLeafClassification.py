
import numpy as np
import sklearn.metrics as metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

import DataModel

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(DataModel.x, DataModel.y, test_size=0.2)


rfClf = RandomForestClassifier()
rfClf.fit(x_train, y_train)
y_pred = rfClf.predict(x_cv)


print("Acc on CV dataset: {0}".format(metrics.accuracy_score(y_cv, y_pred)))

y_test_p = rfClf.predict_proba(DataModel.x_test)
of = DataModel.prepareOutput(y_test_p, DataModel.csvTest.id.values)

of.to_csv('out.csv', index=False)
