import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.pipeline import Pipeline

import numpy as np

from clean_csv import *
from upsampling_predictor import *

print 'Finished loading data'

# first col is ID, last is label
X_train, y_train = data(sys.argv[1])
if len(sys.argv) > 2:
    X_test = testdata(sys.argv[2])
    y_test = None
else:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_train, y_train, test_size=0.1, random_state=0)

    

clf = Pipeline([('Scale', StandardScaler()), ('Classifier', GradientBoostingClassifier(learning_rate=0.05, n_estimators=2000, max_depth=1, subsample=0.3, verbose=2, max_features='auto'))])
reg = Pipeline([('Scale', StandardScaler()), ('Regressor', RandomForestRegressor(n_jobs=8, max_depth=6, n_estimators=900, verbose=2))])

model = UpsamplingPredictor(clf, reg)
model.fit(X_train, X_test, y_train, y_test, bin_thresh=1)

result = model.test()
np.savetxt('predictions_stumps_rf.csv', result, fmt='%d')
