import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, 
    MultiTaskLassoCV, SGDClassifier, SGDRegressor, TheilSenRegressor, 
    RANSACRegressor, HuberRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

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
    name = 'outputs/' + name + '.csv'
    df.to_csv(name)

def genCSV(name, index, prediction):
    '''
    Not a general function, just tacks together the specific case for this Kaggle
    '''
    index = index.A1

    columns = {'lat', 'long'}
    df = pd.DataFrame(prediction, columns=columns, index=index)
    df.index.name = 'index'
    name = 'outputs/' + name + '.csv'
    df.to_csv(name)

def genCSV_predtest(name, prediction, truth):
    predcopy = prediction.copy()
    truthcopy = truth.copy()
    content = np.hstack((prediction, truth))
    name = 'testOutputs/' + name + '.csv'
    np.savetxt(name, content, delimiter=",")

class Models:

    def __init__(self, X_train, X_test, Y_train, Y_test, X_final, index_final):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.X_final = X_final
        self.index_final = index_final

    def ridge(self, name):
        '''
        Ridge
        '''
        sciRidge = Ridge(
            alpha= 300, #tested alpha values, alpha=302 has lowest score
            fit_intercept=True, 
            normalize=False, 
            max_iter=None )
        sciRidge.fit(self.X_train, self.Y_train)
        predict_test = sciRidge.predict(self.X_test)
        MSE = mean_squared_error(predict_test, self.Y_test)
        s = "Sci Ridge              (MSE: %f)" % (MSE)
        print s    
        predict_final = sciRidge.predict(self.X_final)
        genCSV( (name + '_MSE' + str(MSE)), self.index_final, predict_final )

    def ridgeCV(self, name):
        '''
        RidgeCV
        '''
        sciRidgeCV = RidgeCV(
            alphas=(0.001, 0.01, 0.1, 1, 2, 5, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340), #tested alpha values, 321 works best
            fit_intercept=True,
            cv = 11,
            normalize=False )
        sciRidgeCV.fit(self.X_train, self.Y_train)
        predict_test = sciRidgeCV.predict(self.X_test)
        MSE = mean_squared_error(predict_test,self.Y_test)
        s = "Sci RidgeCV            (MSE: %f)" % (MSE)
        print s
        predict_final = sciRidgeCV.predict(self.X_final)
        genCSV( name + '_MSE' + str(MSE), self.index_final, predict_final )

    def lassoCV(self, name):    
        '''
        Lasso Regression
        '''
        sciLasso = MultiTaskLassoCV( 
            fit_intercept=True,
            normalize=False,
            cv=12,
            tol = 0.001 )
        sciLasso.fit(self.X_train, self.Y_train)
        predict_test = sciLasso.predict(self.X_test)
        MSE = mean_squared_error(predict_test,self.Y_test)
        s = "Sci LassoCV            (MSE: %f)" % (MSE)
        print s
        predict_final = sciLasso.predict(self.X_final)
        genCSV( name + '_MSE' + str(MSE), self.index_final, predict_final )

    def SGDClassifier(self):
        '''
        SGD Classifier
        '''
        sciSGDLat = SGDClassifier()
        sciSGDLon = SGDClassifier()
        Y_trainLat = self.Y_train[:,0].A1
        Y_trainLon = self.Y_train[:,1].A1
        Y_trainLat = np.array(["%.2f" % w for w in Y_trainLat])
        Y_trainLon = np.array(["%.2f" % w for w in Y_trainLon])
        
        sciSGDLat.fit(self.X_train, Y_trainLat)
        sciSGDLon.fit(self.X_train, Y_trainLon)

        predict_testLat = sciSGDLat.predict(self.X_test)#.astype(np.float)
        predict_testLon = sciSGDLat.predict(self.X_test)#.astype(np.float)

        predict_test = np.hstack((np.matrix(predict_testLat).transpose(), np.matrix(predict_testLon).transpose()))
        print r2_score(predict_test, self.Y_test)

        # scoreLat = sciSGDLat.score(self.X_test, Y_testLat)
        # scoreLon = sciSGDLon.score(self.X_test, Y_testLon)

        # s = "Sci SGD              (ScoreLat: %f) (ScoreLat: %f)" % (scoreLat, scoreLon)
        # print s    
        #predict_final = sciSGD.predict(self.X_final)
        #genCSV( 'SciSGD.csv', self.index_final, predict_final )

    def SGDRegressor(self):
        '''
        SGD Classifier
        '''
        sciSGDLat = SGDRegressor()
        sciSGDLon = SGDRegressor()
        Y_trainLat = self.Y_train[:,0].A1
        Y_trainLon = self.Y_train[:,1].A1
        Y_testLat = self.Y_test[:,0].A1
        Y_testLon = self.Y_test[:,1].A1
        
        sciSGDLat.fit(self.X_train, Y_trainLat)
        sciSGDLon.fit(self.X_train, Y_trainLon)

        predict_testLat = sciSGDLat.predict(self.X_test)
        predict_testLon = sciSGDLat.predict(self.X_test)

        scoreLat = sciSGDLat.score(self.X_test, Y_testLat)
        scoreLon = sciSGDLat.score(self.X_test, Y_testLat)


        predict_test = np.hstack((np.matrix(predict_testLat).transpose(), np.matrix(predict_testLon).transpose()))


        s = "Sci SGDRegressor     (ScoreLat: %f) (ScoreLon: %f)" % (scoreLat, scoreLon)
        print s



def main():
    trainPredictors = pd.read_csv('InputData/trainPredictors.csv')
    trainTargets = pd.read_csv('InputData/trainTargets.csv')
    testPredictors = pd.read_csv('InputData/testPredictors.csv')
    
    pdIndex = trainPredictors.iloc[:,0]
    pdX = trainPredictors.iloc[:,1:]
    pdY = trainTargets.iloc[:,1:]
    mean = pdY.mean(axis=0)
    std = pdY.std(axis=0)
    pdX.insert(0, "Design", 1.0)

    pdTestIndex = testPredictors.iloc[:,0]
    pdTestX = testPredictors.iloc[:,1:]
    pdTestX.insert(0, "Design", 1.0)
 
    X_orig = np.matrix(pdX.values)
    Y = np.matrix(pdY.values)
    index = np.matrix(pdIndex.values).transpose()
    #print np.unique(np.array(Y[:,0]))
    #print np.unique(np.array(Y[:,1]))

    X_final_orig = np.matrix(pdTestX.values)
    index_final = np.matrix(pdTestIndex.values).transpose()

    '''
    Checking for bad parameters
    '''
    # F=open('blah.txt', 'w')#('parameterCheck(rm5rm7rm30)(0.4).txt', 'a')#

    X = np.hstack((X_orig, np.square(X_orig))) #np.delete(X_orig, [5,7], axis=1)
    X_final = np.hstack((X_final_orig, np.square(X_final_orig))) #np.delete(X_final_orig, [5,7], axis=1)

    for i in range(100):
        print i
        '''
        Training and Testing Data
        '''
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=i)

        '''
        Testing Models
        '''
        models = Models(X_train, X_test, Y_train, Y_test, X_final, index_final)
        #filename = 'Ridge_test0.3_rand' + str(i)  
        #models.ridge(filename)
        #filename = 'RidgeCV_quad_test0.3_rand' + str(i)
        #models.ridgeCV(filename)
        filename = 'LassoCV_quad_test0.3_rand' + str(i)
        models.lassoCV(filename)
        #models.SGD()

    '''
    Plotting
    '''
    #for i in range(X.shape[1]):
    #    plt.scatter(np.array(Y[:,0]),np.array(X[:,i]), alpha=0.1)
    #plt.show()




main()