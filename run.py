import pylab as pl
import numpy as np
from sklearn import linear_model

def load_data(name='train_v2.csv'):
    X = np.loadtxt(open("test.csv","r"), delimiter=",", skiprows=1)
    return X

data = load_data()
X = data[1:len(data)-2]
Y = data[len(data)-2:]
# Split the data into training/testing sets
x_train = X[:len(X)*0.75]
y_test = X[len(X)*0.75:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# The coefficients
print 'Coefficients: \n', regr.coef_
# The mean square error
print ("Residual sum of squares: %.2f" %
        np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print ('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
pl.scatter(diabetes_X_test, diabetes_y_test,  color='black')
pl.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
        linewidth=3)

pl.xticks(())
pl.yticks(())

pl.show()
