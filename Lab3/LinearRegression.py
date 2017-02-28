
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This dataset contains features in a very wide range. Best way to proceed will be to normalize the dataset(with standard deviation). After the data is loaded and normalized, include the intercept term in the dataframe.
df = #Load and preprocess the dataset.


# 1. After you've added the intercept term, define X as the features in the dataframe. Define Y as the target variable.
# 2. Convert them to a numpy array and define beta(the coeffecients) with zeros.
X = 
Y = 
X = np.matrix(X.values)  
Y = np.matrix(Y.values)  
beta = 


# Since we now have every module to calculate our cost function, we'll go ahead and define it.
def costFunction(X, Y, beta):
    '''
    Compute the Least Square Cost Function.
    Return the calculated cost function.
    '''
    return cost


# Define a Gradient Descent method that will update beta in every iteration and also update the cost.
def gradientDescent(X, Y, beta, alpha, iters):
    '''
    Compute the gradient descent function.
    Return beta and the cost array.
    '''
    
    return beta, cost


# Define alpha and number of iterations of your choice and use them to call to gradientDescent function.
#please try different values to see the results, but alpha=0.01 and iters=1000 are suggested.
alpha = #Define  
iters = #Define

result = gradientDescent(X, Y, beta, alpha, iters)


# Implement the Ridge Regression regularization and report the change in coeffecients of the parameters.
def gradientDescentRidge(X, Y, beta, alpha, itersreg, ridgeLambda):
    '''
    Compute the gradient descent function.
    Return beta and the cost array.
    '''
    return beta, cost


# Define alpha, number of iterations and lambda of your choice that minimizes the cost function and use them to call to gradientDescent function. Plot the cost graph with iterations titled "Error vs training" with and without regularization(y axis labeled as cost and x axix labeled as iterations). Then, calculate the MSE.
def MSE(beta):
    '''
    Compute and return the MSE.
    '''
    return mse




#Try differnt values, but the same values for alpha and itersreg are suggested, 0.05 for lambda is suggested
alpha = #Define
itersreg = #Define
ridgeLambda = #Define
regResult = gradientDescentRidge(X, Y, beta, alpha, itersreg, ridgeLambda)
print regResult[0]

#MSE for beta with regularization
beta= #Define
MSE(beta)


#The final result wanted for this portion of the lab is the last plot defined earlier, the explanation regarding the 
#coeffecients of the parameters with Ridge Regression regularization, and the MSE. Please only include the
#MSE in the same .txt file as logistic regression results. Also let the print regResult[0] be there, but do NOT include the 
#outcome for this print in the .txt file. Add the final plot to your report and explain your algorithm, the plot, and 
#the MSE. Generally, what you did in this portion of the lab. Finally, explain the coeffiencts with regularization in your report -PDF file-. 

