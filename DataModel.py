import numpy as np
import pandas as pd
from sklearn import preprocessing

trainDescFile = "..\\Data\\train.csv"
testDescFile = "..\\Data\\test.csv"
trainImagesDir = "..\\Data\\images"

csvRaw = pd.read_csv(trainDescFile)

enc = preprocessing.LabelEncoder()

csvRaw['species_id'] = enc.fit_transform(csvRaw.species).astype(int)

excludedFromTrain = {'id', 'species', 'species_id'}

x = csvRaw[[c for c in csvRaw.columns if c not in excludedFromTrain]].values
y = csvRaw.species_id.values

csvTest = pd.read_csv(testDescFile)

x_test = csvTest[[c for c in csvRaw.columns if c not in excludedFromTrain]].values

def prepareOutput(y, ids):
    assert(y.shape[1] == len(enc.classes_))

    idsX = ids.reshape((ids.shape[0], 1))
    outputDs = pd.DataFrame(data = np.hstack((idsX, y)),
                            columns = np.hstack((["id"], enc.classes_)))
    outputDs.id = outputDs.id.astype(int)

    return outputDs
