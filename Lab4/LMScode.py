import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, confusion_matrix)

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

def LMS(X, Y, iterations=1000, alpha=.0001, batchsize=1000, stop=1.05):
    '''
    LMS Algorithm
    '''
    w = np.matrix(np.zeros(X.shape[1])) #Initialize weights

    cont = True

    #for i in range(0, iterations):
    i = 0
    while (i < iterations) and (cont):

        for k in range(0, X.shape[0], batchsize):

            error = predict(X[k:k+batchsize], w) - Y[k:k+batchsize]

            w = w - (alpha/batchsize) * np.dot(error, X[k:k+batchsize])

            if computeError(X[k:k+batchsize], Y[k:k+batchsize], w) < stop:
                cont = False
                break

        i = i + 1

    return w

def Classify(X, Y, Xtest, Ytest):
    f=open('./perceptron_testfile.csv', 'w+')
    iterations = 100000000

    w = np.matrix(np.ones(X.shape[1]))

    N = X.shape[0]

    k = 0

    for i in range(0, iterations):
        i = k % N

        p = predictClass(X[i], w)[0]
        if ( p > Y[i] ):
            w = w - X[i]
        elif ( p < Y[i] ):
            w = w + X[i]

        if np.array_equal(predictClass(X, w), Y):
            break

        if (i == 0):
            f.write('Iteration: ' + str(k) + '\n')
            print 'Iteration: ' + str(k)
            f.write('Weights: ' + str(w) + '\n')
            print 'Weights: ' + str(w)

            f.write('Training confusion matrix\n')
            f.write(str(confusion_matrix(predictClass(X,w), Y)) + '\n')
            f.write('Testing confusion matrix\n')
            f.write(str(confusion_matrix(predictClass(Xtest,w), Ytest)) + '\n')

            print 'Training confusion matrix'
            print confusion_matrix(predictClass(X,w), Y)
            print 'Testing confusion matrix'
            print confusion_matrix(predictClass(Xtest,w), Ytest)

            f.write('\n')
            print ''

        k = k + 1

    return w

# BEGIN SCRIPT


train = readInData('LMSalgtrain.csv')
test = readInData('LMSalgtest.csv')

Xmean = np.mean(train[0], axis=0)
Xstd = np.std(train[0], axis=0)
X_train = train[0]
X_train = (X_train - Xmean) / Xstd
X_train = np.c_[(np.ones(X_train.shape[0]), X_train)] # Design Matrix

X_test = test[0]
X_test = (X_test - Xmean) / Xstd
X_test = np.c_[(np.ones(X_test.shape[0]), X_test)] # Design Matrix

Y_train = train[1]
Y_test = test[1]

Z_train = train[2]
Z_test = test[2]


file=open('./testfile.csv', 'w+')

beta = OLSData(X_train, Y_train)
Ein = computeError(X_train, Y_train, beta)
Eout = computeError(X_test, Y_test, beta)
print "Beta: " + str(beta)
print "Ein OLS MSE:  " + str(Ein)
print "Eout OLS MSE: " + str(Eout)

alpha = [.001, .0001, .00001]
batch = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
stop_condition = [1.1, 1.05, 1.04, 1.03, 1.02, 1.01]

print "alpha, batchsize, stop_condition, calculated_beta, train_MSE, test_MSE"
file.write("alpha, batchsize, stop_condition, calculated_beta, train_MSE, test_MSE,\n")
for a in alpha:
    for b in batch:
        for s in stop_condition:
            LMSbeta = LMS(X_train, Y_train, 1000, a, b, s)
            LMSEin = computeError(X_train, Y_train, LMSbeta)
            LMSEout = computeError(X_test, Y_test, LMSbeta)
            
            output = str(a)+ ', ' + str(b) + ', ' + str(s) + ', ' + str(LMSbeta.round(decimals=1)) + ', ' + str(LMSEin) + ', ' + str(LMSEout) + ',\n'
            print output
            file.write(output)


CLASSbeta = Classify(train[0], train[2], test[0], test[2])
print "Final Beta: " + str(CLASSbeta)
