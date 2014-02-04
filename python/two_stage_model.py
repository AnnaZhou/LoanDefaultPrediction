# General approach for a two-stage approach to this problem
from python.clean_csv import *

class TwoStageRegressor:
    
    def __init__(self, train_filename, test_filename, clf=None, reg=None):
        self.X_train, self.y_train = data(train_filename)
        self.X_test = testdata(test_filename) 

	self.clf = clf
	self.reg = reg

    def fit(self):
	self.clf.fit(self.X_train, self.y_train)
	
    	
