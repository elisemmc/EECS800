
# coding: utf-8



from sklearn import datasets
import pandas as pd
import numpy as np
from math import sqrt
import operator

#Please complete the following KNN code based on the given instructions


# Load the training and testing data from the provided files into pandas dataframes.




#Please handle the data with 'dataframe' type, remember to print/display the test and training datasets
train = pd.read_csv('KnnTrain.csv')
test = pd.read_csv('KnnTest.csv')

print 'Training Data'
print train
print ''

print 'Test Data'
print test
print ''


# 1. Define k as the number of unique elements in the target column.
# 2. Define a prediction list.
# 3. Define a groundTruth list that saves the target values from the test dataset.



k = len(train['Target'].unique())#define
prediction = []
groundTruth = test['Target']

print k


# Now define a eucledian distance function that returns the distance between two instances.



def eucledianDistance(instance_1, instance_2):
    '''
    This function takes in 2 instances and returns the Eucledian Distance between them.
    '''
    return eucDistance


# Now that we have the eucledian distance, we can use it collect the k most similar instances for a given unseen instance.<br/>
# This is a straight forward process of calculating the distance for all instances and selecting a subset with the smallest distance values.<br/>
# Define the classDistance function that will return the k most similar neighbors from a training dataset for an instance(using the already defined euclideanDistance function).<br/>
# The function should return a list that has k instances of class variable and eucledian distance,like this for k=3:<br/>
# (1,0.32)<br/>
# (0,0.35)<br/>
# (0,0.37)<br/>



def classDistance(df_train, instance_test, k):
    '''
    This function takes in training dataset, a test instance and k.
    It should return the top k values for the least Eucledian distances for each instance along with class variable.
    '''
    return classDist[:k]


# Now that we have k least distances, we can make the prediction for the test instance. We can do this by allowing each neighbor to vote for their class attribute, and take the majority vote as the prediction.



def voting(classDist):
    '''
    This function takes in the result from the classDistance function and returns the label after voting.
    '''
    print votes
    return votes['''?''']['''?''']


# Calculate the accuracy of your model. The function should take the prediction and groundTruth as inputs and return the 
# confusion matrix. The confusion matrix is of 'dataframe' type.



def confusionMatrix(prediction, groundTruth):
    '''
    This function returns and print the confusion matrix which is of type 'dataframe'.
    '''
    return confMatrix


# Now, test your model based on the test dataset. Add your prediction to the 'prediction' list.



for x in range(len(test)):
    cd = classDistance(train,test.iloc[x], k)
    prediction.append(voting(cd))


# Call the confusionMatrix function and print the confusion matrix as well as the accuracy of the model.



groundTruth = pd.Series(groundTruth)
prediction = pd.Series(prediction)
conf = confusionMatrix(groundTruth, prediction)
print conf
accuracy = 1#define accuracy
print 'Accuracy = '+str(accuracy*100)+'%'





