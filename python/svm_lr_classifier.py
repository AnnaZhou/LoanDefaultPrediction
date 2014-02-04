import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from scipy.stats import threshold
from sklearn.metrics import roc_auc_score as auc

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

#write_data('X_train-svm-rf.csv', X_train)
#write_data('y_train-svm-rf.csv', y_train)

model = LinearSVC(C=0.01, penalty="l1", dual=False, verbose=2, class_weight={0: 1, 1: 2})
model.fit(X_train, y_train)
#joblib.dump(model, 'svm_cl.pkl')

# Features have been selected by SVM
X_train = model.transform(X_train)
X_test = model.transform(X_test)

# first col is ID, last is label
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
model.fit(X_train, y_train)
joblib.dump(model, 'random_forest_cl.pkl')


def thresh(y, p):
    y = threshold(y, threshmax=p, newval=1)
    y = threshold(y, threshmin=p, newval=0)
    return y

print 'Finished fitting the model'
prob = model.predict_proba(X_test)
y_prob = prob[:, 1:]
for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    score = auc(y_test, thresh(y_prob, t))
    print t, 'auc', score
    
