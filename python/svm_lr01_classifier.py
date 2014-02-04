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
X_train, y_train = X, y

#write_data('X_train-svm-rf.csv', X_train)
#write_data('y_train-svm-rf.csv', y_train)

model = LinearSVC(C=0.01, penalty="l1", dual=False, verbose=2, class_weight={0: 1, 1: 2})
model.fit(X_train, y_train)
#joblib.dump(model, 'svm_cl.pkl')

# Features have been selected by SVM
X_train = model.transform(X_train)

# first col is ID, last is label
model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
model.fit(X_train, y_train)
print 'Finished fitting the model'

def thresh(y, p):
    y = threshold(y, threshmax=p, newval=1)
    y = threshold(y, threshmin=p, newval=0)
    return y

X_test = testdata(sys.argv[2])
prob = model.predict_proba(X_test)
y_prob = prob[:, 1:]
for i,d in enumerate(y_prob):
    if d > 0.1:
        y_prob[i] = 8.0
    else:
        y_prob[i] = 0

np.savetxt('predictions', y_prob) 
