import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import datasets
from sklearn.grid_search import GridSearchCV

import numpy as np

from utils import *

data = getData(sys.argv[1])
print 'Finished loading data'

# first col is ID, last is label
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:, 1:-1], data[:, -1:] , test_size=0.15, random_state=0)

model = RandomForestClassifier(n_jobs=-1, max_depth=20, n_estimators=900, verbose=2)
model.fit(X_train, y_train)
print 'Finished fitting the model'

result = model.score(X_test, y_test)
print 'Results:'
print result
