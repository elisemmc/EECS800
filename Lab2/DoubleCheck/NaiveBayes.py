
# coding: utf-8

# In[6]:
from __future__ import division
from sklearn import datasets
import pandas as pd
import numpy as np
import math
import operator

#Please complete the following Naive Bayes code based on the given instructions


# Load the training and test dataset.

#Please handle the data with 'dataframe' type, remember to print/display the test and training datasets
train = pd.read_csv('NaiveBayesTrain.csv')
test = pd.read_csv('NaiveBayesTest.csv')

print 'Training Data'
print train
print ''

print 'Test Data'
print test
print ''

groundtruth = test['target']


# We can use a Gaussian function to estimate the probability of a given attribute value, 
# given the known mean and standard deviation for the attribute estimated from the training data.
# Knowing that the attribute summaries where prepared for each attribute and class value, 
# the result is the conditional probability of a the attribute value given a class value.

def probCalculator(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calcConditionalProb(testset,df_temp):
    '''
    This function takes the test dataset and a dataframe given one class label.
    The function returns the conditional probability given a column(feature).
    '''
    #hint: you can test.ix[:,i]
    #prob = 1.0
    d = {}

    means = df_temp.mean()
    stdevs = df_temp.std()

    for index,row in testset.iterrows():
        prob = 1.0
        for i in testset:
            # print probCalculator(row[i], means[i], stdevs[i]) 
            prob *= probCalculator(row[i], means[i], stdevs[i])
            # prob[i] = probCalculator(row[i], means[i], stdevs[i])
        d[index] = prob
    return d


# Follow intructions for the given code snippet:

# In[ ]:

prob_df = pd.DataFrame()
#define a variable probTarget which is equal to the probablity of a given class.(upto 4 decimal places)
# print len(test[test['target']==1])
test = test.drop('target', axis = 1)
rowCount = len(train.index)
probZero = len(train[train['target']==0])/rowCount
probOne = len(train[train['target']==1])/rowCount
probTarget = { 1 : probOne , 0 : probZero }
# print probTarget


# For each label in the training dataset, we compute the probability of the test instances.

condProbs = {}

for label in train['target'].unique():
    df_temp = train[train['target']==label]
    df_temp = df_temp.drop('target', axis = 1)
    testset = test.copy(deep=True)
    condProbs = calcConditionalProb(testset, df_temp)
    condProbs.update((k, v * probTarget[label]) for k,v in condProbs.items())
    prob_df[label] = condProbs.values()

print 'Probability DataFrame'
print prob_df.round(4)
print ''

# Define a list 'prediction' that stores the label of the class with highest probability for each test 
# instance which are stored in prob_df dataframe.

prediction = []

for index,row in prob_df.iterrows():
    prediction.append( 1 if (row[1] > row[0]) else 0 )

print 'Prediction'
print prediction
print ''


# Calculate the accuracy of your model. The function should take the prediction and groundTruth as inputs and return the 
# confusion matrix. The confusion matrix is of 'dataframe' type.


def confusionMatrix(prediction, groundTruth):
    '''
    Return and print the confusion matrix.
    '''
    conf = pd.crosstab(groundTruth, prediction, rownames=['Actual'], colnames=['Predicted'])

    print 'Confusion Matrix'
    print conf
    print ''
    return conf

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



groundTruth =  pd.Series(groundtruth)
prediction = pd.Series(prediction)
conf = confusionMatrix(prediction, groundTruth)
accuracy = accuracy(conf) #( conf[0][0] + conf[1][1] ) / ( conf[0][0] + conf[0][1] + conf[1][0] + conf[1][1] ) #define accuracy
print 'Accuracy = '+str(accuracy*100)+'%'
print ''

