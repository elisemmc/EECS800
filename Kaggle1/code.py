import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

class linearReg:

    def __init__(self, trainingFeatures, groundTruth):
        '''
        Initialize linear regression model with training data
        '''
        numFeatures = trainingFeatures.shape[1]

        self.alpha = 0.0001
        self.ridgeLambda = 0.9
        self.iterations = 1000

        self.trainingFeatures = trainingFeatures
        self.beta = np.zeros((1,numFeatures))
        self.Ymean = np.zeros((1,groundTruth.shape[1]))

    def setAlpha(self, newAlpha):
        self.alpha = newAlpha

    def setLambda(self, newLambda):
        self.ridgeLambda = newLambda

    def zeroBeta(self):
        self.beta = np.zeros(self.beta.shape)

    def centerY(self, Y):
        centeredY = Y

        for i in range(Y.shape[1]):
            self.Ymean[0,i] = Y[:,i].mean()

        for i in range(Y.shape[0]): 
            centeredY[i,:] = Y[i,:] - self.Ymean
        
        return centeredY

    def uncenterY(self, Y):
        uncenteredY = Y
        
        for i in range(Y.shape[0]):
            uncenteredY[i,:] = Y[i,:] + self.Ymean

        return uncenteredY

    # Since we now have every module to calculate our cost function, we'll go ahead and define it.
    def costFunction(self, X, Y, beta):
        '''
        Compute the Least Square Cost Function.
        Return the calculated cost function.
        '''
        element = np.power( ( ( X*beta.T ) - Y ), 2 )
        cost = np.sum( element ) / ( 2 * len(X) )

        return cost

    # Define a Gradient Descent method that will update beta in every iteration and also update the cost.
    def gradientDescent(self, X, Y):
        '''
        Compute the gradient descent function.
        Return beta and the cost array.
        '''
        cost = []

        m = len(X)

        beta = self.beta
        parameters = len(X.T)

        for i in range(0, self.iterations):
            loss = ( X * beta.T ) - Y

            for j in range(parameters):
                gradient =  np.multiply(loss, X[:,j])
                beta[0,j] = beta[0,j] - ((self.alpha/m) * np.sum(gradient))

            cost.append( self.costFunction(X, Y, beta) )

        return beta, cost

    def hypothesis(beta, X):
        temp = np.multiply(beta, X.transpose())
        return temp

    def gradient(beta, X, Y):
        '''
        This function returns the gradient calucated.
        '''
        grad = np.dot(hypothesis(beta,X)-Y, X)

        return grad

    # Implement the Ridge Regression regularization and report the change in coeffecients of the parameters.
    def gradientDescentRidge(self, X, Y, beta):
        '''
        Compute the gradient descent function.
        Return beta and the cost array.
        '''
        cost = []

        m = len(X)

        if len(beta) == 0:
            beta = self.beta

        parameters = len(X.T)

        for i in range(0, self.iterations):
            loss = ( X * beta.T ) - Y

            for j in range(parameters):
                # Tried removing outliers....didn't work
                # Xtemp = np.multiply(X[:,j], (X[:,j] > -4).astype(int))
                # Xtemp = np.multiply(X[:,j], (X[:,j] < 4).astype(int))

                gradient =  np.multiply(loss, X)
                
                if j == 0:
                    beta[0,j] = beta[0,j] - ( (self.alpha/m) * np.sum(gradient) )
                else:
                    beta[0,j] = beta[0,j] - ( (self.alpha/m) * np.sum(gradient) + self.ridgeLambda * (beta[0,j])/m)

            cost.append( self.costFunction(X, Y, beta) )

        return beta, cost

    def MSE(self, X, Y):
        '''
        Compute and return MSE
        '''
        element = np.power((X - Y), 2)
        mse = np.sum(element)/ len(X)

        return mse

    def predict(self, X, beta):
        prediction = np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            prediction[i,0] = np.dot( X[i,:], beta[0,:].transpose() )

        return prediction

    def plotGraph(self, costs):
        plt.title("Error vs training")
        plt.ylabel("cost")
        plt.xlabel("iterations")
        for i in costs:
            plt.plot(i)
        #plt.legend()
        plt.show()

def genCSV4(name, index, latitude, longitude):
    '''
    Not a general function, just tacks together the specific case for this Kaggle
    '''
    result = np.zeros((index.shape[0], 2))
    index = index.A1
    result[:,0] = np.array(latitude[:,0])
    result[:,1] = np.array(longitude[:,0])

    columns = {'lat', 'long'}
    df = pd.DataFrame(result, columns=columns, index=index)
    df.index.name = 'index'
    name = 'outputs/' + name
    df.to_csv(name)

def genCSV(name, index, prediction):
    '''
    Not a general function, just tacks together the specific case for this Kaggle
    '''
    index = index.A1

    columns = {'lat', 'long'}
    df = pd.DataFrame(prediction, columns=columns, index=index)
    df.index.name = 'index'
    name = 'outputs/' + name
    df.to_csv(name)

def main():
    trainPredictors = pd.read_csv('InputData/trainPredictors.csv')
    trainTargets = pd.read_csv('InputData/trainTargets.csv')
    testPredictors = pd.read_csv('InputData/testPredictors.csv')
    
    pdIndex = trainPredictors.iloc[:,0]
    pdX = trainPredictors.iloc[:,1:]
    pdY = trainTargets.iloc[:,1:]
    pdX.insert(0, "Design", 1.0)

    pdTestIndex = testPredictors.iloc[:,0]
    pdTestX = testPredictors.iloc[:,1:]
    pdTestX.insert(0, "Design", 1.0)

    
    X = np.matrix(pdX.values)
    Y = np.matrix(pdY.values)
    index = np.matrix(pdIndex.values).transpose()

    testX = np.matrix(pdTestX.values)
    testIndex = np.matrix(pdTestIndex.values).transpose()

    model = linearReg(X,Y)

    cY = model.centerY(Y)

    Xtr = X[:400,:]
    Xte = X[400:,:]

    Ytr = Y[:400,:]
    cYtr = Y[:400,:]
    Yte = Y[400:,:]

    # print "alpha:      %f" % model.alpha
    # print "iterations: %d" % model.iterations
    # print ""

    # lat = model.gradientDescent(Xtr, cYtr[:,0])
    # lon = model.gradientDescent(Xtr, cYtr[:,1])

    # print "OLS"
    # print "Latitude MSE:  %f" % model.MSE(model.predict(Xte, lat[0]) + model.Ymean[0,0], Yte )
    # print "Longitude MSE: %f" % model.MSE(model.predict(Xte, lon[0]) + model.Ymean[0,1], Yte )
    # print ""

    '''
    # This is my RidgeRegression
    latBeta = model.beta
    lonBeta = model.beta

    latMSE = model.MSE(model.predict(Xte, latBeta) + model.Ymean[0,0], Yte )
    lonMSE = model.MSE(model.predict(Xte, lonBeta) + model.Ymean[0,1], Yte )

    for i in range(78):
        prevLatMSE = latMSE
        prevLonMSE = lonMSE

        latRidge = model.gradientDescentRidge(Xtr, cYtr[:,0], latBeta)
        lonRidge = model.gradientDescentRidge(Xtr, cYtr[:,1], lonBeta)

        #make sure to recenter the y's !!!
        latMSE = model.MSE(model.predict(Xte, latRidge[0]) + model.Ymean[0,0], Yte )
        lonMSE = model.MSE(model.predict(Xte, lonRidge[0]) + model.Ymean[0,1], Yte )

        if latMSE < prevLatMSE:
            latBeta = latRidge[0]

        if lonMSE < prevLonMSE:
            lonBeta = lonRidge[0]

        if ( ( latMSE > prevLatMSE ) and ( lonMSE > prevLonMSE ) ):
            break

        print "Ridge (iterations: %f)" % (i*model.iterations)
        print "Lambda: %f    Latitude MSE: %f    Longitude MSE: %f" % (model.ridgeLambda, latMSE, lonMSE)
        print ""

    genCSV4( 'Ridge_a0.0001_iter78000_l0.9.csv', testIndex, model.predict(testX, latBeta), model.predict(testX, lonBeta) )
    '''


    ####
    '''
    https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/linear_model/ridge.py#L1105
    '''
    sciLinReg = LinearRegression()
    sciLinReg.fit(X, Y[:,0])
    predict0 = sciLinReg.predict(Xte)
    sciLinReg.fit(X, Y[:,1])
    predict1 = sciLinReg.predict(Xte)

    print "Sci LinearRegression"
    print "Latitude MSE:  %f" % model.MSE(predict0, Yte )
    print "Longitude MSE: %f" % model.MSE(predict1, Yte )
    print ""

    predict0 = sciLinReg.predict(testX)
    predict1 = sciLinReg.predict(testX)
    genCSV4( 'SciLinReg.csv', testIndex, predict0, predict1 )     

    # sciRidge = Ridge()
    # sciRidge.fit(Xtr, Ytr)
    # predict = sciRidge.predict(Xte)

    # print "Sci Ridge"
    # print "Latitude MSE:  %f" % model.MSE(predict, Yte )
    # print "Longitude MSE: %f" % model.MSE(predict, Yte )
    # print ""

    # predict = sciLinReg.predict(testX)
    # genCSV( 'SciRidge.csv', testIndex, predict ) 

    # sciRCV = RidgeCV()
    # sciRCV.fit(Xtr, Ytr)
    # predict = sciRCV.predict(Xte)

    # print "Sci Ridge CV"
    # print "Latitude MSE:  %f" % model.MSE(predict, Yte )
    # print "Longitude MSE: %f" % model.MSE(predict, Yte )
    # print ""

    # predict = sciRCV.predict(testX)
    # genCSV( 'SciRidgeCV.csv', testIndex, predict )


    # model.genCSV( 'Ridge_testing.csv', testIndex, model.predict(testX, latRidge[0]), model.predict(testX, lonRidge[0]))

    # costs = []

    # costs.append(lat[1])
    # costs.append(lon[1])

    # costs.append(latRidge[1])
    # costs.append(lonRidge[1])

    # model.plotGraph(costs)
    
    # i = 16
    # plt.plot(Y[:,0],Y[:,1],'o')
    # plt.plot(Y[:,1],X[:,i],'o')
    # plt.show()

main()
#calls the main() method at program start
#this can be done if you just like the idea of scripting it, but this is easier to read and edit if the program gets bigger
#if  __name__ =='__main__':main()