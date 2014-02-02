import sys

from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.grid_search import GridSearchCV

import numpy as np

def getData():
    with open(sys.argv[1], 'r') as f:
        arr = np.loadtxt(f)
        return arr

data = getData()

iris = datasets.load_iris()
model = GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=100, subsample=0.5)


#scores = cross_validation.cross_val_score(model, iris.data, iris.target, cv=5)
#print scores

X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

param_grid = [
        {'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001], 'n_estimators': [50, 100, 200, 500, 800, 1500, 2000, 5000], 'subsample': [0.3, 0.5, 0.7, 1.0]}
        ]
print 
mod = GridSearchCV(GradientBoostingRegressor(loss='ls'), param_grid, cv=5)
mod.fit(X_train, y_train)

print 'Best parameter set found: '
print 
print mod.best_estimator_
print
print 'Grid scores on development set:'
print
for params, mean_score, scores in mod.grid_scores_:
    print '%0.3f (+/-%0.03f) for %r' % (mean_score, scores.std()/2, params)


