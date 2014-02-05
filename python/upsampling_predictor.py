import numpy as np
from harness import TwoModelPredictor

class UpsamplingPredictor(TwoModelPredictor):

    def fit(self, X_train, X_test, y_train, y_test=None, bin_thresh=0):
        """
        Upsample the training data before calling superclass method.
        By default upsample until classes are roughly equal, else create r
        copies of each row from the smaller class.
        """
	print 'Fitting the UpsamplingPredictor'
        y_train_bin = self.threshold(y_train, bin_thresh)

        zeros = np.bincount(y_train_bin.astype(int))[0] 
        ones = np.bincount(y_train_bin.astype(int))[1]
        print zeros, ones

        # Get only the rows having nonzero label
        Xy = np.transpose(np.vstack((np.transpose(X_train), y_train)))
	print 'Xy'
	print Xy[:5, :]
        Xy_nz = Xy[np.logical_or.reduce([Xy[:,-1] > 0])]
	print 'Xy_nz'
	print Xy_nz[:5, :]
	
	Xy_nz_chunk = Xy_nz
	count = ones
        while count < zeros:
	    count += ones
            Xy_nz = np.vstack((Xy_nz, Xy_nz_chunk))
            print 'ones', count

        Xy_all = np.vstack((Xy_nz, Xy))
        X, y = Xy_all[:, :-1], Xy_all[:, -1]
	print 'X'
	print X[:5, :]
	print 'y'
	print y[:5]
        TwoModelPredictor.fit(self, X, y, X_test, y_test, bin_thresh=bin_thresh)

        
