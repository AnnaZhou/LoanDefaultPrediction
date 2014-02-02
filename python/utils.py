import numpy as np

def getData(path):
    with open(path, 'r') as f:
        return np.loadtxt(f, delimiter=',')
