
# coding: utf-8


import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxint

import warnings
warnings.filterwarnings("ignore")

#Please complete the following Expectation Maximization code, implementing a batch EM is sufficient for this lab

#set a random seed, remember to set the correct seed if you want to use another command for seeding
rand.seed(124)

# we have *two* clusters. Note that the covariance matrices are diagonal
mu = [0, 6]
sig = [ [3, 0], [0, 4] ]

muPrime = [6, 0]
sigPrime = [ [5, 0], [0, 2] ]

#Generate samples of type MVN and size 100 using mu/sigma and muPrime/sigmaPrime. 
num_points = 100
x1, y1 = ( [rand.gauss(mu[0], sig[0][0]) for _ in range(num_points)], [rand.gauss(mu[1], sig[1][1]) for _ in range(num_points)] )
x2, y2 = ( [rand.gauss(muPrime[0], sigPrime[0][0]) for _ in range(num_points)], [rand.gauss(muPrime[1], sigPrime[1][1]) for _ in range(num_points)] ) 

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
  prob = alpha
  for v in range(len(val)): prob *= norm.pdf(val[v],mu[v],sig[v][v])
  return prob 

# The E-step, estimate w, this w is the "soft guess" step for the class labels. You have to use the already defined posteriors in this step.
def expectation(dataFrame, parameters):
  '''This function uses the posteriors to estimate w.'''
  values = dataFrame[['x','y']].as_matrix()
  alpha = parameters['alpha'].tolist()
  mu = (parameters['mu'].tolist(), parameters['muPrime'].tolist())
  sig = (parameters['sig'].tolist(), parameters['sigPrime'].tolist())
  label = np.zeros(values.shape[0])
  for v in range(values.shape[0]): label[v] = 0 if ( posterior(values[v],mu[0],sig[0],alpha[0])  > posterior(values[v],mu[1],sig[1],alpha[1]) ) else 1
  dataFrame['label'] = label
  return dataFrame #dataframe with the soft guess for the labels

# The M - step: update estimates of alpha, mu, and sigma
def maximization(dataFrame, parameters):
  '''Update parameters'''
  new_params = parameters.copy()
  c1_df = dataFrame[ dataFrame['label'] == 0 ];c2_df = dataFrame[ dataFrame['label'] == 1 ]
  new_params['alpha'] = [ float(len(c1_df)) / float(len(dataFrame)), float(len(c2_df)) / float(len(dataFrame)) ]
  new_params['mu'] = [ c1_df['x'].mean(), c1_df['y'].mean() ]
  new_params['muPrime'] = [ c2_df['x'].mean(), c2_df['y'].mean() ]
  new_params['sig'] = [ [ c1_df['x'].std(), 0 ], [ 0, c1_df['y'].std() ] ]
  new_params['sigPrime'] = [ [ c2_df['x'].std(), 0 ], [ 0, c2_df['y'].std() ] ]
  return new_params

# Check Convergence, define your convergence criterion. You can define a new function for this purpose or just check it in the loop. You will have to use this function at the end of each while/for loop's EM iteration to check whether we have reached "convergence" or not. So to test for convergence, we can calculate the log likelihood at the end of each EM step (e.g. model fit with these parameters) and then test whether it has changed “significantly” (defined by the user, e.g. it should be something similar to: if(loglik.diff < 1e-6) ) from the last EM step. If it has, then we repeat another step of EM. If not, then we consider that EM has converged and then these are our final parameters.

def convergeCheck(params, new_params, epsilon):
  sig_old = [params['sig'].tolist()[0][0], params['sig'].tolist()[1][1], params['sigPrime'].tolist()[0][0], params['sigPrime'].tolist()[1][1]] 
  sig_new = [new_params['sig'].tolist()[0][0], new_params['sig'].tolist()[1][1], new_params['sigPrime'].tolist()[0][0], new_params['sigPrime'].tolist()[1][1]]
  mu_dist =  np.linalg.norm( np.array(params['mu'].tolist() + params['muPrime'].tolist()) - np.array(new_params['mu'].tolist() + new_params['muPrime'].tolist()) )
  sig_dist = np.linalg.norm( np.array(sig_old) - np.array(sig_new) )
  return ( (mu_dist < epsilon) and (sig_dist < epsilon) )
# Iterate until convergence: with E-step, M-step, checking our etimates of mu/checking whether we have reached convergece, and updating the parameters for the next iteration. This part of the code should print a figure for *each* iteration, the *final* parameters, and #iterations. The final outcome that you have to submit is your EM code and a .pdf report. Your report should have the plots for **each** iteration, your **explanation on the convergence criterion you used based on last paragraph's explanations**, your final parameters, and the general flow of your code.

# loop until the parameters converge

iters = 0
params = pd.DataFrame(initialGuess)
epsilon = 0.001
max_iters = 20
converge = False
old_df = df
old_params = params

while ( ( iters < max_iters ) and ( converge == False ) ):
  iters += 1
  # E-step
  new_df = expectation(old_df, old_params)
  # M-step
  new_params = maximization(new_df, old_params)
  # see if our estimates of mu have changed and check convergence
  converge = convergeCheck(old_params.copy(), new_params.copy(), epsilon)
  # print parameters for each iteration
  print new_params
  # update labels and parameters for the next iteration
  if new_params.isnull().values.any():
    print '\n\n \033[93mFAILED TO CLASSIFY -- PLEASE RERUN \n\n'
    break
  else:  
    old_df = new_df
    old_params = new_params
  # return a scatter plot for each iteration, e.g. plt.scatter(df_new['x'], df_new['y'], ...) while the colors are based on the labels
  f = plt.figure()
  plt.scatter(new_df['x'],new_df['y'],c=new_df['label'],s=100)
  f.savefig("iter{}.png".format(iters))
