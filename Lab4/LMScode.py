import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def predict(X, beta):
    '''
    return array of predictions given X and beta
    '''
    prediction = np.array([(x*beta.T).item() for x in X])

    return prediction

def predictClass(X, beta):
    prediction = np.array([(x*beta.T).item() for x in X])
    prediction = [ ( 1 if ( x > 0 ) else 0 ) for x in prediction ]
    return prediction
# Since we now have every module to calculate our cost function, we'll go ahead and define it.
def computeError(X, Y, beta):
    '''
    compute and return MSE
    '''
    p = predict(X, beta)
    element = np.power( ( p - Y ), 2 )
    MSE = np.sum( element ) / ( len(p) )

    return MSE

def readInData(filename):
    df = pd.read_csv(filename, dtype=np.float64) #Load and preprocess the dataset.

    cols = df.shape[1]

    X = df.iloc[:,0:cols-2]

    Y = df.iloc[:,cols-2:cols-1]

    Z = df.iloc[:,cols-1:cols]

    X = np.matrix(X.values)
    Xmean = np.mean(X, axis=0)
    Xstd = np.std(X, axis=0)
    X = (X - Xmean) / Xstd
    X = np.c_[(np.ones(X.shape[0]), X)] # Design Matrix
    
    Y = np.matrix(Y.values).A1
    # Ymean = np.mean(Y, axis=0)
    # Y = (Y - Ymean)

    Z = np.matrix(Z.values).A1

    return X, Y, Z

def OLSData(data, labels):
    X = data
    Y = labels

    model = smf.OLS(Y,X)
    result = model.fit()
    beta = result.params

    return np.matrix(beta)

def LMS(X, Y):
    '''
    LMS Algorithm
    '''
    alpha = 0.0001
    iterations = 1000

    w = np.matrix(np.zeros(X.shape[1])) #Initialize weights
    w_new = w
    print X.shape

    for i in range(0, iterations):
        error = predict(X, w) - Y

        w_new = w - alpha * np.dot(error, X)

        w = w_new

    return w

def Classify(X, Y):
    alpha = 1
    iterations = 100

    X = np.c_[(np.ones(X.shape[0]), X)]

    w = np.matrix(np.zeros(X.shape[1]))

    N = X.shape[0]

    k = 0

    for i in range(0, iterations):
        i = k % N

        p = predictClass(X[i], w)[0]
        if ( p != Y[i] ):
            w = w + Y[i] * X[i]  

        if predictClass(X, w) == Y[i]:
            break

        k = k + 1

    return w

# BEGIN SCRIPT


readIn = readInData('LMSalgtrain.csv')

X_train, X_test, Y_train, Y_test = train_test_split( readIn[0], readIn[1], test_size=0.3, random_state=0)

beta = OLSData(X_train, Y_train)
Ein = computeError(X_train, Y_train, beta)
Eout = computeError(X_test, Y_test, beta)
print "Beta: " + str(beta)
print "Ein OLS MSE:  " + str(Ein)
print "Eout OLS MSE: " + str(Eout)

LMSbeta = LMS(X_train, Y_train)
LMSEin = computeError(X_train, Y_train, LMSbeta)
LMSEout = computeError(X_test, Y_test, LMSbeta)
print "Beta: " + str(LMSbeta)
print "Ein LMS MSE:  " + str(LMSEin)
print "Eout LMS MSE: " + str(LMSEout)

X_train, X_test, Y_train, Y_test = train_test_split( readIn[0], readIn[2], test_size=0.3, random_state=0)

Classify(X_train, Y_train)
