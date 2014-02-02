import sys
import numpy as np
import pylab as P

def getData():
    with open(sys.argv[1], 'r') as f:
        arr = np.loadtxt(f)
        return arr

data = getData()

counts, scores = map(lambda x: x[0], data[:, :1].tolist()), map(lambda x: -1*x[0], data[:, 1:].tolist())
print counts[:10]
print scores[:10]

def ext(ls1, ls2):
    ls1.extend(ls2)
    return ls1

ls = reduce(lambda x,y: ext(x, y), [[s for i in xrange(int(c))] for s,c in zip(scores, counts)])
P.hist(ls, bins=100, normed=True)
P.show()
