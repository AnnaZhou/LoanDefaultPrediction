import sys

from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

import numpy as np

from utils import *

data = getData(sys.argv[1])

# first col is ID, last is label
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:, 1:-1], data[:, -1:] , test_size=0.15, random_state=0)

model = LinearSVC(C=0.01, penalty="l1", dual=False, verbose=2)
model.fit(X_train, y_train)
joblib.dump(model, 'svm_cl.pkl')

result = model.score(X_test, y_test)
print result

report = classification_report(y_test, model.predict(X_test))
print 'Report:'
print report
