import sys

from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

import numpy as np

from utils import *

data = getData(sys.argv[1])

# first col is ID, last is label
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:, 1:-1], data[:, -1:] , test_size=0.15, random_state=0)

model = GradientBoostingClassifier(max_depth=20, learning_rate=0.01, n_estimators=300, subsample=0.3, verbose=2)
model.fit(X_train, y_train)
joblib.dump(model, 'boost_cl.pkl')

result = model.score(X_test, y_test)
print result

report = classification_report(y_test, model.predict(X_test))
print 'Report:'
print report
