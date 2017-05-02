
# coding: utf-8

import numpy as np


# You have 3 tasks to do:
# First, with the help of the weight vector, implement the *Support* function which will provide us with the indeces of support vectors.
# Second, with the help of the weight vector,  implement the  *Slack* function which will return all vectors' indeces with non-zero slack.
# Third, given the Lagrange multiplier, implement the *weightVector* function for returning the corresponding weight vector. 
# You have to submit your code and a *short* report, .pdf, explaning your results which are the requested indeces as well as the general flow of the functions.
# 
# 
# 
# For given vectors x, y, and alpha, we would want to compute the primal weight vector.
def weightVector(x, y, LagrangeMultiplier):
    '''
    This method should return the primal weight vector.
    '''
    w = np.zeros(x.shape[1])

    for i in range(x.shape[0]):
    	w += LagrangeMultiplier[i] * y[i] * x[i]

    return w


# For a given weight vector we want to now implement the find support function(The tolerance is 0.0015). This function returns the indices of the support vectors.
def Support(x, y, w, b, tol=0.0015):
    '''
    This method should return the indices for all of the support vectors.
    '''
    support = set()
    gammas = np.zeros(x.shape[0])

    xMag = np.linalg.norm(w,ord=1)

    for i in range(x.shape[0]):
    	gammas[i] = abs( ( np.dot(w, x[i]) + b ) / xMag )

    minimum = min(gammas)

    for i in range(x.shape[0]):
    	if ( ( gammas[i] <= minimum + tol ) and ( gammas[i] >= minimum - tol ) ):
    		support.add(i)
    
    return support


#  The Slack function which will provide us with the indices of the slack vectors.
def Slack(x, y, w, b):
    '''
    This function should return the indices for all of the slack vectors, given a primal support vector instance.
    '''
    slack = set()

    for i in range(x.shape[0]):
    	if not( y[i] * ( np.dot( w, x[i] ) + b ) >= 1 ):
    		slack.add(i)

    return slack



Inseparable = np.array([(2, 10, +1),
               (8, 2, -1),
               (5, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (7, 1, -1),
               (5, 2, -1)])

Separable = np.array([(-2, 2, +1),    
              (0, 8, +1),     
              (2, 1, +1),     
              (-2, -3, -1),   
              (0, -1, -1),    
              (2, -3, -1),    
              ])




#you don't have to add anything to the code from here on except printing the requested indeces
x1 = Separable[:, 0:2]
y1 = Separable[:, 2]
x2 = Inseparable [:, 0:2]
y2 = Inseparable [:, 2]


# Given the values of alphas, call the weightVector and store the result in 'w'.
LagrangeMultiplier = np.zeros(len(x1))
LagrangeMultiplier[4] = 0.34
LagrangeMultiplier[0] = 0.12
LagrangeMultiplier[2] = 0.22


w = weightVector(x1, y1, LagrangeMultiplier)

#Given b, find the support set.
b = -0.2
s = Support(x1, y1, w, b)
print s

# Given the following values, compute the slack indeces.
w2 = [-.25, .25]
b2 = -.25
s1 = Slack(x2, y2, w2, b2)
print s1

