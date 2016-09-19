import numpy as np
import pandas as pd
from sklearn import preprocessing

trainDescFile = "..\\Data\\train.csv"
trainImagesDir = "..\\Data\\images"

csvRaw = pd.read_csv(trainDescFile)

enc = preprocessing.LabelEncoder()

csvRaw['species_id'] = enc.fit_transform(csvRaw.species).astype(int)

excludedFromTrain = {'id', 'species', 'species_id'}

x = csvRaw[[c for c in csvRaw.columns if c not in excludedFromTrain]].values
y = csvRaw.species_id.values
