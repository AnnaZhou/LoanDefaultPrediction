import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor

def getData():
    with open(sys.argv[1], 'r') as f:
        arr = np.loadtxt(f)
        return arr


data = getData()
train = data[:450]
test = data[450:]

forest = RandomForestRegressor(n_estimators=100, verbose=10)

forest = forest.fit(train[:, 1:], train[:, :1])

correct = test[:, :1]
output = forest.score(test[:, 1:], correct)
print output

