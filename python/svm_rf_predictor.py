import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.pipeline import Pipeline

import numpy as np

from clean_csv import *
from upsampling_predictor import *

X, y = data(sys.argv[1])
print 'Finished loading data'

# first col is ID, last is label
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.15, random_state=0)

clf = Pipeline([('Scale', StandardScaler()), ('SVM', LinearSVC(C=0.01, penalty="l1", dual=False, verbose=2))])
reg = Pipeline([('Scale', StandardScaler()), ('Random Forest', RandomForestRegressor(n_jobs=-1, max_depth=6, n_estimators=900, verbose=2))])

model = UpsamplingPredictor(clf, reg)
model.fit(X_train, X_test, y_train, y_test)

result = model.test()
