import numpy as np
from harness import TwoModelPredictor

class UpsamplingPredictor(TwoModelPredictor):

    def fit(self, X_train, X_test, y_train, y_test=None, r=-1):
        """
        Upsample the training data before calling superclass method.
        By default upsample until classes are roughly equal, else create r
        copies of each row from the smaller class.
        """
	print 'Fitting the UpsamplingPredictor'
        y_train_bin = self.threshold(y_train, 0)
        if r == -1:
            r = np.bincount(y_train_bin.astype(int))[0] / np.bincount(y_train_bin.astype(int))[1] - 1
	print 'r =', r
        # Get only the rows having nonzero label
        Xy = np.transpose(np.vstack((np.transpose(X_train), y_train)))
        Xy_nz = Xy[np.logical_or.reduce([Xy[:,-1] > 0])]
        while r > 0:
	    print r, 'iterations to go'
            r -= 1
            Xy_nz = np.vstack((Xy_nz, Xy_nz))
        Xy_all = np.vstack((Xy_nz, Xy))
        X, y = Xy_all[:, :-1], Xy_all[:, -1]
        TwoModelPredictor.fit(self, X, y, X_test, y_test)

        
