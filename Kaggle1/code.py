import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Functions:

    def __init__(self, trainingFeatures, groundTruth):
        '''
        Initialize linear regression model with training data
        '''
        numFeatures = df.shape[1]

        self.trainingFeatures = trainingFeatures
        self.beta = np.matrix(np.zeros(numFeatures))
        self.alpha = 0.01
        self.ridgeLambda = 0.05
        self.iterations = 1000

    def setAlpha(self, newAlpha):
        self.alpha = newAlpha

    def setLambda(self, newLambda):
        self.ridgeLambda = newLambda

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
    def gradientDescent(self, X, Y, beta, alpha):
        '''
        Compute the gradient descent function.
        Return beta and the cost array.
        '''
        cost = []

        m = len(X)

        beta_temp = np.matrix(np.zeros(beta.shape))
        parameters = len(X.T)

        for i in range(0, self.iters):
            loss = ( X * beta.T ) - Y

            for j in range(parameters):
                gradient =  np.multiply(loss, X[:,j])
                beta_temp[0,j] = self.beta[0,j] - ((self.alpha/m) * np.sum(gradient))

            beta = beta_temp
            cost.append( costFunction(X, Y, beta) )

        return beta, cost

    # Implement the Ridge Regression regularization and report the change in coeffecients of the parameters.
    def gradientDescentRidge(self, X, Y, beta, alpha, itersreg, ridgeLambda):
        '''
        Compute the gradient descent function.
        Return beta and the cost array.
        '''
        cost = []

        m = len(X)

        beta_temp = np.matrix(np.zeros(beta.shape))
        parameters = len(X.T)

        for i in range(0, iters):
            loss = ( X * beta.T ) - Y

            for j in range(parameters):
                gradient =  np.multiply(loss, X[:,j])
                if j == 0:
                    beta_temp[0,j] = beta[0,j] - ((alpha/m) * np.sum(gradient))
                else:
                    beta_temp[0,j] = beta[0,j] - ( (alpha/m) * np.sum(gradient) + ridgeLambda * (beta[0,j])/m)

            beta = beta_temp
            cost.append( costFunction(X, Y, beta) )

        return beta, cost

    # Define alpha, number of iterations and lambda of your choice that minimizes the cost function and use them to 
    # call to gradientDescent function. Plot the cost graph with iterations titled "Error vs training" with and without 
    # regularization(y axis labeled as cost and x axix labeled as iterations). Then, calculate the MSE.
    def MSE(self, beta):
        '''
        Compute and return the MSE.
        '''
        element = np.power( ( X*beta.T - Y ), 2 )
        mse = np.sum( element ) / ( len(X) )

        return mse

def main():
    #predictors = pd.read_csv('InputData/trainPredictors.csv', dtype=np.float64)
    df = pd.read_csv('HousingData_LinearRegression.csv', dtype=np.float64) #Load and preprocess the dataset.
    for i in df:
        mean = df[i].mean()
        std = df[i].std()
        df[i] = ( df[i] - mean )/std
    df.insert(0, "Design", 1.0)

    cols = df.shape[1]
    X = df.iloc[:,0:cols-1]
    Y = df.iloc[:,cols-1:cols]

    X = np.matrix(X.values)
    Y = np.matrix(Y.values)
    
    model = Functions(predictors)










#calls the main() method at program start
#this can be done if you just like the idea of scripting it, but this is easier to read and edit if the program gets bigger
if  __name__ =='__main__':main()