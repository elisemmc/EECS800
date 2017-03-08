import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, MultiTaskLassoCV, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.kernel_ridge import KernelRidge

def MSE(X, Y):
    '''
    Compute and return MSE
    '''
    element = np.power((X - Y), 2)
    mse = np.sum(element)/ len(X)

    return mse

def genCSV4(name, index, latitude, longitude):
    '''
    Not a general function, just tacks together the specific case for this Kaggle
    '''
    result = np.zeros((index.shape[0], 2))
    index = index.A1
    result[:,0] = np.array(latitude[:,0])
    result[:,1] = np.array(longitude[:,0])

    columns = {'lat', 'long'}
    df = pd.DataFrame(result, columns=columns, index=index)
    df.index.name = 'index'
    name = 'outputs/' + name
    df.to_csv(name)

def genCSV(name, index, prediction):
    '''
    Not a general function, just tacks together the specific case for this Kaggle
    '''
    index = index.A1

    columns = {'lat', 'long'}
    df = pd.DataFrame(prediction, columns=columns, index=index)
    df.index.name = 'index'
    name = 'outputs/' + name
    df.to_csv(name)

def genCSV_predtest(name, prediction, truth):
    predcopy = prediction.copy()
    truthcopy = truth.copy()
    content = np.hstack((prediction, truth))
    name = 'testOutputs/' + name
    np.savetxt(name, content, delimiter=",")

def main():
    trainPredictors = pd.read_csv('InputData/trainPredictors.csv')
    trainTargets = pd.read_csv('InputData/trainTargets.csv')
    testPredictors = pd.read_csv('InputData/testPredictors.csv')
    
    pdIndex = trainPredictors.iloc[:,0]
    pdX = trainPredictors.iloc[:,1:]
    pdY = trainTargets.iloc[:,1:]
    pdX.insert(0, "Design", 1.0)

    pdTestIndex = testPredictors.iloc[:,0]
    pdTestX = testPredictors.iloc[:,1:]
    pdTestX.insert(0, "Design", 1.0)
 
    X_orig = np.matrix(pdX.values)
    Y = np.matrix(pdY.values)
    index = np.matrix(pdIndex.values).transpose()

    X_final_orig = np.matrix(pdTestX.values)
    index_final = np.matrix(pdTestIndex.values).transpose()

    '''
    Checking for bad parameters
    '''
    F=open('blah.txt', 'w')#('parameterCheck(rm5rm7rm30)(0.4).txt', 'a')#
    #for i in range(X_orig.shape[1]):
        # print i
        # if i==(5 or 7):
        #     i=i+1
    X = np.delete(X_orig, [5,7], axis=1)
    X_final = np.delete(X_final_orig, [5,7], axis=1)

    #F.write(str(i)+'\n')

    s_Ridge = 0#.204114
    s_RidgeCV = 0#.203701
    s_LassoCV = 0#.199943

    thresh = 0.0001
    '''
    Training and Testing Data
    '''
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.5, random_state=17)

    '''
    Ridge
    '''
    sciRidge = Ridge(
        alpha= 300, #tested alpha values, alpha=302 has lowest score
        fit_intercept=True, 
        normalize=False, 
        max_iter=None )
    sciRidge.fit(X_train, Y_train)
    predict_test = sciRidge.predict(X_test)
    score = sciRidge.score(X_test, Y_test)
    if score >= s_Ridge + thresh:
        s = "Sci Ridge              (Score: %f)" % (score)
        print s
        F.write(s + '\n')
    # predict_final = sciRidge.predict(X_final)
    # genCSV( 'SciRidge.csv', index_final, predict_final )

    '''
    RidgeCV
    '''
    sciRidgeCV = RidgeCV(
        alphas=(0.001, 0.01, 0.1, 1, 2, 5, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340), #tested alpha values, 321 works best
        fit_intercept=True,
        cv = 11,
        normalize=False )
    sciRidgeCV.fit(X, Y)
    predict_test = sciRidgeCV.predict(X_test)
    score = sciRidgeCV.score(X_test,Y_test)
    if score >= s_RidgeCV + thresh:
        s = "Sci RidgeCV            (Score: %f)" % (score)
        print s
        F.write(s + '\n')
    predict_final = sciRidgeCV.predict(X_final)
    genCSV( 'SciRidgeCV.csv', index_final, predict_final )

    '''
    Lasso Regression
    '''
    sciLasso = MultiTaskLassoCV( 
        fit_intercept=True,
        normalize=False,
        cv=12,
        tol = 0.01 )
    sciLasso.fit(X_train, Y_train)
    predict_test = sciLasso.predict(X_test)
    score = sciLasso.score(X_test,Y_test)
    if score >= s_LassoCV + thresh:
        s = "Sci LassoCV            (Score: %f)" % (score)
        print s
        F.write(s + '\n')
    # predict_final = sciLasso.predict(X_final)
    # genCSV( 'sciLasso.csv', index_final, predict_final )




main()