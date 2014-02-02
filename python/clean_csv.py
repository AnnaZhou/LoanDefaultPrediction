import sys
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

# First pass over the data: average all columns.
# Second pass: replace each NA with the average for that column

totals = [0 for i in xrange(800)]
counts = [0 for i in xrange(800)]

# Replace NA cols with column mean
def testdata(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)

    X = np.asarray(X.values, dtype=float)

    col_mean = stats.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    data = np.asarray(X[1:,1:-3], dtype=float)

    return data

def data(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)

    X = np.asarray(X.values, dtype = float)

    col_mean = stats.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds]=np.take(col_mean, inds[1])

    labels = np.asarray(X[1:,-1], dtype=float)
    data = np.asarray(X[1:,1:-4], dtype=float)
    return data, labels

def normalize(X):
    normalizer = StandardScaler(copy=False)
    return normalizer.fit_transform(X)
