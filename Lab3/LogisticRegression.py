
# coding: utf-8



import numpy as np
import pandas as pd
import scipy.optimize as opt
from __future__ import division


# Load the LogisticData.csv for this assignment and check if it has loaded successfully. In this excercise, we will use the same dataset for training and testing. After training the model on the best beta, we will see how well does our model perform.
df = #define


# Next, preprocess the data. X should contain only the predictors, Y shuould be the reponse variable and beta should be a vector with length = the number of features and value = 0.
X =   
Y = 
groundTruth = df['Y']
#Normalize only if you are using gradient descent. Apply standard deviation for normalization.

X = np.matrix(X.values)  
Y = np.matrix(Y.values)
beta = 


# Define a sigmoid function and return the value tht has been calculated for z
def sigmoid(z):
    '''
    Here sigmoid value of 'z' needs to be returned.
    '''
    return sig


# Define the cost function for Logistic Regression. Remember to calculate the sigmoid values as well.
def costFunction(beta, X, Y):
    '''
    This function returns the value computed from the cost function.
    '''
    return result


# Define a gradient function that takes in beta, X and Y as parameters and returns the best betas and cost. 
def gradientDescent(X, Y, beta, alpha, iters):
    '''
    Compute the gradient descent function.
    Return the newly computed beta and the cost array.
    '''

    return beta, cost


# Try out multiple values of 'alpha' and 'iters' so that you get the optimum result.
#please try different values to see the results, but alpha=0.01 and iters=10000 are suggested.
alpha = #define
iters = #define
result = gradientDescent(X, Y, beta, alpha, iters)


# Now , only define the gradient function that we can use in the SciPy's optimize module to find the optimal betas. 
def gradient(beta, X, Y):
    '''
    This function returns the gradient calucated.
    '''
    #for i in range(parameters):
    #####
    #grad[i] =
    return grad


# Optimize the parameters given functions to compute the cost and the gradients. We can use SciPy's optimization to do the same thing.
# Define a variable result and complete the functions by adding the right parameters.

#the optimised betas are stored in the first index of the result variable
result = opt.fmin_tnc(func= , x0= , fprime = , args= )


# Define a predict function that returns 1 if the probablity of the result from the sigmoid function is greater than 0.5, using the best betas and 0 otherwise.

def predict(beta, X): 
    '''
    This function returns a list of predictions calculated from the sigmoid using the best beta.
    '''
    return #define


# Store the prediction in a list after calling the predict function with best betas and X.
bestBeta = np.matrix(result[0])  
predictions = predict(bestBeta, X)


# Calculate the accuracy of your model. The function should take the prediction and groundTruth as inputs and return the 
# confusion matrix. The confusion matrix is of 'dataframe' type.
def confusionMatrix(prediction, groundTruth):
    '''
    Return the computed confusion matrix.
    '''
    return pd.crosstab(groundTruth, prediction, rownames=['True'], colnames=['Predicted'], margins=False)


# Call the confusionMatrix function and print the confusion matrix as well as the accuracy of the model.
#The final outputs that we need for this portion of the lab are conf and acc. Copy conf and acc in a .txt file.
#Please write a SHORT report and explain these results. Include the explanations for both logistic and linear regression
#in the same PDF file. 

groundTruth = pd.Series(groundTruth)
prediction = pd.Series(predictions)
conf = #define
acc = #define
print 'Accuracy = '+str(acc)+'%'

