from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import threshold
import numpy as np

class TwoModelPredictor:

    def __init__(self, clf, reg):
        self.clf = clf
        self.reg = reg

    def threshold(self, y, p):
        """
        Threshold y st values < p => 0, and values > p => 1.
        """
        mi = threshold(y, threshmin=p+1, newval=0)
        #print np.bincount(mi.astype(int))
        ma = threshold(mi, threshmax=p, newval=1)
        #print np.bincount(ma.astype(int))
        return ma

    def fit(self, X_train, y_train, X_test, y_test=None, bin_thresh=0):
        """
        Fit the classifier with the training data and show the AUC score.
        Then train the regressor using all training data having a nonzero label.
        """
        self.X_train, self.y_train, = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
	
        y_train_bin = self.threshold(y_train, bin_thresh)
        print 'bincount'
        print np.bincount(y_train.astype(int))
	print 'bincount (bin)'
        print np.bincount(y_train_bin.astype(int))
        self.clf.fit(X_train, y_train_bin)
        if y_test != None:
            auc_score = auc(self.threshold(y_test, bin_thresh), self.clf.predict(X_test))
        print 'Classifier AUC =', auc_score
        
        # Get only the rows having nonzero label
        Xy = np.transpose(np.vstack((np.transpose(X_train), y_train)))
        Xy_nz = Xy[np.logical_or.reduce([Xy[:,-1] > 0])]
        X_train_nz = Xy_nz[:, :-1]
        y_train_nz = Xy_nz[:, -1]

        self.reg.fit(X_train_nz, y_train_nz)

    def predict(self, X):
        """
        First use the classifier to predict default or not. Then use the regressor
        to predict the quantity of default for every sample. Finally zero out the 
        prediction for samples where the classifier predicted no default.
        """
        zpred = self.clf.predict(X)
        zeros = np.where(zpred == 0)

        nzpred = self.reg.predict(X)
        nzpred[zeros] = 0

        return nzpred

    def test(self):
        result = pred = self.predict(self.X_test)
        
        if self.y_test != None:
            err = mae(self.y_test, pred)
            print 'MAE =', err
        else:
            # Add the ids
            result = np.transpose(np.vstack((np.transpose(self.X_test[:, 0]), pred)))
        return result

