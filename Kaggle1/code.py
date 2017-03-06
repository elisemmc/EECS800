import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class linearRegression:

    def __init__(self, trainingFeatures, groundTruth):
        '''
        Initialize linear regression model with training data
        '''
        numFeatures = trainingFeatures.shape[1]

        self.alpha = 0.001
        self.ridgeLambda = 0.1
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

    # Implement the Ridge Regression regularization and report the change in coeffecients of the parameters.
    def gradientDescentRidge(self, X, Y):
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
                if j == 0:
                    beta[0,j] = beta[0,j] - ((self.alpha/m) * np.sum(gradient))
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

    def genCSV(self, name, index, latitude, longitude):
        '''
        Not a general function, just tacks together the specific case for this Kaggle
        '''
        result = np.zeros((index.shape[0], 2))
        index = index.A1
        result[:,0] = np.array(latitude[:,0])
        result[:,1] = np.array(longitude[:,0])
        result = self.uncenterY(result)

        columns = {'lat', 'long'}
        df = pd.DataFrame(result, columns=columns, index=index)
        df.index.name = 'index'
        df.to_csv(name)

        return result

    def plotGraph(self, costs):
        plt.title("Error vs training")
        plt.ylabel("cost")
        plt.xlabel("iterations")
        for i in costs:
            plt.plot(i)
        #plt.legend()
        plt.show()

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

    model = linearRegression(X,Y)

    cY = model.centerY(Y)

    lat = model.gradientDescent(X,cY[:,0])
    lon = model.gradientDescent(X,cY[:,1])

    latRidge = model.gradientDescentRidge(X, cY[:,0])
    lonRidge = model.gradientDescentRidge(X, cY[:,1])

    print model.MSE(model.predict(X, latRidge[0]) + model.Ymean[0,0], Y )
    print model.MSE(model.predict(X, lonRidge[0]) + model.Ymean[0,1], Y )

    #model.genCSV( 'OLS_a0.01_i1000.csv', testIndex, model.predict(testX, lat[0]), model.predict(testX, lon[0]) )
    model.genCSV( 'Ridge_a0.001_i1000_l0.1.csv', testIndex, model.predict(testX, latRidge[0]), model.predict(testX, lonRidge[0]))

    # costs = []

    # costs.append(lat[1])
    # costs.append(lon[1])

    # costs.append(latRidge[1])
    # costs.append(lonRidge[1])

    # model.plotGraph(costs)

main()
#calls the main() method at program start
#this can be done if you just like the idea of scripting it, but this is easier to read and edit if the program gets bigger
#if  __name__ =='__main__':main()