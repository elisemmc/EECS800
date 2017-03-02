
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This dataset contains features in a very wide range. 
# Best way to proceed will be to normalize the dataset(with standard deviation). 
# After the data is loaded and normalized, include the intercept term in the dataframe.
df = pd.read_csv('HousingData_LinearRegression.csv', dtype=np.float64) #Load and preprocess the dataset.
for i in df:
    mean = df[i].mean()
    std = df[i].std()
    df[i] = ( df[i] - mean )/std
df.insert(0, "Design", 1.0)

# print df
# 1. After you've added the intercept term, 
# define X as the features in the dataframe. Define Y as the target variable.
# 2. Convert them to a numpy array and define beta(the coeffecients) with zeros.
#X = df.drop(df.columns[len(df.columns)-1], axis=1 )
cols = df.shape[1]
X = df.iloc[:,0:cols-1]
Y = df.iloc[:,cols-1:cols]
# for i in X:
#     mean = X[i].mean()
#     std = X[i].std()
#     X[i] = ( X[i] - mean )/std

#Y = df.iloc[:,-1]
# mean = Y.mean(axis=0)
# std = Y.std(axis=0)
# Y = (Y-mean)

X = np.matrix(X.values)
Y = np.matrix(Y.values)
beta = np.matrix(np.zeros(cols-1))

# Since we now have every module to calculate our cost function, we'll go ahead and define it.
def costFunction(X, Y, beta):
    '''
    Compute the Least Square Cost Function.
    Return the calculated cost function.
    '''
    element = np.power( ( ( X*beta.T ) - Y ), 2 )
    cost = np.sum( element ) / ( 2 * len(X) )

    return cost

#print costFunction (X, Y, beta)

# Define a Gradient Descent method that will update beta in every iteration and also update the cost.
def gradientDescent(X, Y, beta, alpha, iters):
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
            beta_temp[0,j] = beta[0,j] - ((alpha/m) * np.sum(gradient))

        beta = beta_temp
        cost.append( costFunction(X, Y, beta) )

    return beta, cost

# Define alpha and number of iterations of your choice and use them to call to gradientDescent function.
#please try different values to see the results, but alpha=0.01 and iters=1000 are suggested.
alpha = 0.01 #Define  
iters = 1000 #Define

result = gradientDescent(X, Y, beta, alpha, iters)

# Implement the Ridge Regression regularization and report the change in coeffecients of the parameters.
def gradientDescentRidge(X, Y, beta, alpha, itersreg, ridgeLambda):
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
def MSE(beta):
    '''
    Compute and return the MSE.
    '''
    element = np.power( ( X*beta.T - Y ), 2 )
    mse = np.sum( element ) / ( len(X) )

    return mse

#Try differnt values, but the same values for alpha and itersreg are suggested, 0.05 for lambda is suggested
alpha = alpha #Define
itersreg = iters #Define
ridgeLambda = 0.05 #Define
beta.fill(0)
regResult = gradientDescentRidge(X, Y, beta, alpha, itersreg, ridgeLambda)

plt.title("Error vs training")
plt.ylabel("cost")
plt.xlabel("iterations")
plt.plot(result[1], color="red", label="regular")
plt.plot(regResult[1], color="blue", label="ridge")
plt.legend()

#MSE for beta with regularization
beta = beta#Define
print MSE(beta)

plt.show()

#The final result wanted for this portion of the lab is the last plot defined earlier, the explanation regarding the 
#coeffecients of the parameters with Ridge Regression regularization, and the MSE. Please only include the
#MSE in the same .txt file as logistic regression results. Also let the print regResult[0] be there, but do NOT include the 
#outcome for this print in the .txt file. Add the final plot to your report and explain your algorithm, the plot, and 
#the MSE. Generally, what you did in this portion of the lab. Finally, explain the coeffiencts with regularization in your report -PDF file-. 

