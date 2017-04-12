
# coding: utf-8


import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sys import maxint

#Please complete the following Expectation Maximization code, implementing a batch EM is sufficient for this lab

#set a random seed, remember to set the correct seed if you want to use another command for seeding
rand.seed(124)

# we have *two* clusters. Note that the covariance matrices are diagonal
mu = [0, 6]
sig = [ [3, 0], [0, 4] ]

muPrime = [6, 0]
sigPrime = [ [5, 0], [0, 2] ]

#Generate samples of type MVN and size 100 using mu/sigma and muPrime/sigmaPrime. 
x1, y1 = np.random.multivariate_normal(mu, sig, 100).T #MVN sample of size 100 using mu and sigma
x2, y2 = np.random.multivariate_normal(muPrime, sigPrime, 100).T #MVN sample of size 100 using muPrime and sigmaPrime

x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))

#Convert the data which now includes 'x' and 'y' to a dataframe. 
data = {'x': x, 'y': y}
df = pd.DataFrame(data=data)


# Load the data and inspect it
df.head()


# Initial guess for mu, sigma, and alpha which are initially bad, α = probability of class asignment. ∑α=1 for k=1 to K.
initialGuess = { 'mu': [2,2], 'sig': [ [1, 0], [0, 1] ], 'muPrime': [5,5], 'sigPrime': [ [1, 0], [0, 1] ], 'alpha': [0.4, 0.6]}


# Compute the posterior with the help of computing the pdf, e.g. using norm.pdf
def posterior(val, mu, sig, alpha):
  '''posteriors'''
  probs = np.zeros( ( val.shape[0], len(alpha) ) )
  num_classes = len(alpha)
  sums = 0

  for c in range(num_classes):
  	a = alpha[c]
   	m = mu[c]
   	s = sig[c]
	pdf =  multivariate_normal.pdf(x=val, mean=m, cov=s)
	sums += a * pdf
	probs[:,c] = a*pdf

  for c in range(num_classes):
  	probs[:,c] = np.divide( probs[:,c], sums )
  
  print probs
  #print np.sum(probs, axis=1)
  return probs 

# The E-step, estimate w, this w is the "soft guess" step for the class labels. You have to use the already defined posteriors in this step.
def expectation(dataFrame, parameters):
  '''This function uses the posteriors to estimate w.'''

  data = dataFrame.as_matrix()
  alpha = parameters['alpha'].tolist()
  
  mu_1 = parameters['mu'].tolist()
  sig_1 = parameters['sig'].tolist()

  mu_2 = parameters['muPrime'].tolist()
  sig_2 = parameters['sig'].tolist()

  mu = (mu_1, mu_2)
  sig = (sig_1, sig_2)

  posterior(data, mu, sig, alpha)
  
  # print multivariate_normal.pdf(x=data[0], mean=mu_1, cov=sig_1)
  

  return #dataframe with the soft guess for the labels


# The M - step: update estimates of alpha, mu, and sigma
def maximization(dataFrame, parameters):
  '''Update parameters'''
  return parameters


# Check Convergence, define your convergence criterion. You can define a new function for this purpose or just check it in the loop. You will have to use this function at the end of each while/for loop's EM iteration to check whether we have reached "convergence" or not. So to test for convergence, we can calculate the log likelihood at the end of each EM step (e.g. model fit with these parameters) and then test whether it has changed “significantly” (defined by the user, e.g. it should be something similar to: if(loglik.diff < 1e-6) ) from the last EM step. If it has, then we repeat another step of EM. If not, then we consider that EM has converged and then these are our final parameters.

# Iterate until convergence: with E-step, M-step, checking our etimates of mu/checking whether we have reached convergece, and updating the parameters for the next iteration. This part of the code should print a figure for *each* iteration, the *final* parameters, and #iterations. The final outcome that you have to submit is your EM code and a .pdf report. Your report should have the plots for **each** iteration, your **explanation on the convergence criterion you used based on last paragraph's explanations**, your final parameters, and the general flow of your code.



# loop until the parameters converge

iters = 0
params = pd.DataFrame(initialGuess)

while iters < 1:
  iters += 1

  expectation(df, params)
  # E-step
 
  # M-step
 
  # see if our estimates of mu have changed and check convergence
  
  # print parameters for each iteration

  # update labels and parameters for the next iteration
   
  # return a scatter plot for each iteration, e.g. plt.scatter(df_new['x'], df_new['y'], ...) while the colors are based on the labels

