
# coding: utf-8


from __future__ import division
import numpy as np
import pandas as pd
import scipy.optimize as opt

# Load the LogisticData.csv for this assignment and check if it has loaded successfully. In this excercise, we will use the same dataset for training and testing. After training the model on the best beta, we will see how well does our model perform.
df = pd.read_csv('LogisticData.csv', dtype=np.float64)#define


# Next, preprocess the data. X should contain only the predictors, Y shuould be the reponse variable and 
# beta should be a vector with length = the number of features and value = 0.
X = df.drop(df.columns[len(df.columns)-1], axis=1 )
for i in X:
    mean = X[i].mean(axis=0)
    std = X[i].std(axis=0)
    X[i] = ( X[i] - mean )/std
X.insert(0, "Design", 1.0)
Y = df.iloc[:,-1]  
groundTruth = Y.copy() #df['Y']

#Normalize only if you are using gradient descent. Apply standard deviation for normalization.
X = np.matrix(X.values)  
Y = np.matrix(Y.values)
beta = np.matrix(np.zeros((1,X.shape[1])))

# Define a sigmoid function and return the value that has been calculated for z
def sigmoid(z):
    '''
    Here sigmoid value of 'z' needs to be returned.
    '''
    sig = 1 / ( 1 + np.exp(-z) )

    return sig

def hypothesis(beta, X):
    temp = np.dot(beta, X.transpose())
    return sigmoid(temp)

# Define the cost function for Logistic Regression. Remember to calculate the sigmoid values as well.
def costFunction(beta, X, Y):
    '''
    This function returns the value computed from the cost function.
    '''
    m = len(X)

    costSum = 0
    for i in range(m):
        hypothesis = 1 / ( 1 + np.exp(- np.sum( np.dot(X[i,:], beta.T) )) )
        costSum += Y[:,i] * np.log( hypothesis ) + ( 1-Y[:,i] ) * ( np.log( 1 - hypothesis ) )
    cost = - costSum
    
    return cost

# # Now , only define the gradient function that we can use in the SciPy's optimize module to find the optimal betas. 
def gradient(beta, X, Y):
    '''
    This function returns the gradient calucated.
    '''
    grad = np.dot(hypothesis(beta,X)-Y, X)

    return grad

# Define a gradient function that takes in beta, X and Y as parameters and returns the best betas and cost. 
def gradientDescent(X, Y, beta, alpha, iters):
    '''
    Compute the gradient descent function.
    Return the newly computed beta and the cost array.
    '''

    cost = np.zeros(iters)

    for i in range(iters):
        beta = beta - alpha * gradient(beta,X,Y)
        cost[i] = costFunction(beta, X, Y)

    return beta, cost

# Try out multiple values of 'alpha' and 'iters' so that you get the optimum result.
#please try different values to see the results, but alpha=0.01 and iters=10000 are suggested.
alpha = 0.01 #define
iters = 10000 #define
result = gradientDescent(X, Y, beta, alpha, iters)
print "gradient descent beta"
print result[0]

# Optimize the parameters given functions to compute the cost and the gradients. We can use SciPy's optimization to do the same thing.
# Define a variable result and complete the functions by adding the right parameters.

#the optimised betas are stored in the first index of the result variable
result = opt.fmin_tnc(func = costFunction, x0 = beta, fprime = gradient, args=(X,Y))


# Define a predict function that returns 1 if the probablity of the result from the sigmoid function is greater than 0.5, using the best betas and 0 otherwise.

def predict(beta, X): 
    '''
    This function returns a list of predictions calculated from the sigmoid using the best beta.
    '''
    h = np.array(hypothesis(beta,X))
    m = len(X)
    prediction = np.zeros(m)

    for i in range(m):
        prediction[i] = ( 1 if (h[0,i] > 0.5) else 0 )

    return prediction#define

# Store the prediction in a list after calling the predict function with best betas and X.
bestBeta = np.matrix(result[0])

# Calculate the accuracy of your model. The function should take the prediction and groundTruth as inputs and return the 
# confusion matrix. The confusion matrix is of 'dataframe' type.
def confusionMatrix(prediction, groundTruth):
    '''
    Return the computed confusion matrix.
    '''
    return pd.crosstab(groundTruth, prediction, rownames=['True'], colnames=['Predicted'], margins=False)

def accuracy(confusionMatrix):
    '''
    Calculates accuracy given the confusion matrix
    '''
    numerator = 0
    denominator = 0

    for index, row in confusionMatrix.iterrows():
        for i in range(len(row)):
            if(index == i):
                numerator += row[i]

            denominator += row[i]

    return numerator / denominator

# Call the confusionMatrix function and print the confusion matrix as well as the accuracy of the model.
#The final outputs that we need for this portion of the lab are conf and acc. Copy conf and acc in a .txt file.
#Please write a SHORT report and explain these results. Include the explanations for both logistic and linear regression
#in the same PDF file. 

print "best beta"
print bestBeta
predictions = predict(bestBeta, X)

conf = confusionMatrix(predictions, np.array(groundTruth)) #define
print conf
acc = accuracy(conf) #define
print 'Accuracy = '+str(acc*100)+'%'

