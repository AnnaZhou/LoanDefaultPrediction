import sys


from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn import datasets
from sklearn.grid_search import GridSearchCV

import numpy as np
from clean_csv import *

X, y = data(sys.argv[1])
for i,d in enumerate(y):
    y[i] = 1 if d > 0 else 0
mean = np.mean(y)
print 'Mean(labels) = ', mean

X = normalize(X)
print 'Finished loading data'

# first col is ID, last is label
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)


param_grid = [
	{'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 7}],
	 'C': [0.01, 0.1]}
        ]
print 

for score in ['precision', 'recall', 'f1']:
	mod = GridSearchCV(LinearSVC(penalty="l1", dual=False, verbose=2), param_grid, cv=2, scoring=score)
	mod.fit(X_train, y_train)

	print 'Best parameter set found: '
	print 
	print mod.best_estimator_
	print
	print 'Grid scores on development set:'
	print
	for params, mean_score, scores in mod.grid_scores_:
	    print '%0.3f (+/-%0.03f) for %r' % (mean_score, scores.std()/2, params)

	y_true, y_pred = y_test, mod.predict(X_test)
	print classification_report(y_true, y_pred)


