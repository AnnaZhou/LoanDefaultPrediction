from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import threshold

class TwoModelPredictor:

    def __init__(self, clf, reg):
        self.clf = clf
        self.reg = reg

    def threshold(self, y, p):
        """
        Threshold y st values < p => 0, and values > p => 1.
        """
        return threshold(threshold(y, threshmin=p, newval=0), threshmax=p, newval=1)

    def fit(self, X_train, X_test, y_train, y_test=None):
        """
        Fit the classifier with the training data and show the AUC score.
        Then train the regressor using all training data having a nonzero label.
        """
        self.X_train, self.y_train, = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        y_train_bin = self.threshold(y_train)
        self.clf.fit(X_train, y_train_bin)
        auc_score = auc(y_test, self.clf.predict(X_test))
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
        zeros = np.where(pred == 0)

        nz = pred[np.logical_or.reduce([pred[:,-1] == 1])]
        nzpred = self.reg.predict(X)
        nzpred[zeros] = 0

        return nzpred

    def test(self):
        result = pred = self.predict(self.X_test)
        
        if self.y_test:
            err = mae(self.y_test, pred)
            print 'MAE =', err
        else:
            # Add the ids
            result = np.transpose(np.vstack((np.transpose(self.X_test[:, 0]), pred)))
        return result

