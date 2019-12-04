from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.externals import joblib

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import os
import warnings

warnings.filterwarnings('ignore')


# --------------------------------load data---------------------------------

def loadData(file):
    f = open(file, "r")
    X = []
    for l in f.readlines():
        l = l.split(' ')
        _X = []
        for i in range(len(l)):
            if l[i].isdigit():
                _X.append(int(l[i]))
        X.append(_X)
    return mat(X)


# --------------------------------error statistic--------------------------


def statistic(_Y, Y):
    Error = 0
    Sum = 0
    errNum1 = [0] * 10
    errNum2 = [0] * 10
    sumNum1 = [0] * 10
    sumNum2 = [0] * 10
    m = Y.shape[0]
    for i in range(m):
        if int(_Y[i, 0]) != int(Y[i, 0]):
            errNum1[int(Y[i, 0])] += 1
            errNum2[int(_Y[i, 0])] += 1
            Error += 1
        sumNum1[int(Y[i, 0])] += 1
        sumNum2[int(Y[i, 0])] += 1
        Sum += 1
    print('------------------------------------')
    print('error statistic: ')
    print('sum digits: ', Sum)
    print('accuracy: ', round((1.0 - Error / Sum) * 100.0, 3), '%')
    for i in range(10):
        print(i, ': ', end='\t')
        print(round(errNum1[i] / sumNum1[i] * 100.0, 2), '%', end='\t\t')
        print(round(errNum2[i] / sumNum2[i] * 100.0, 2), '%')
    print('------------------------------------')


# ---------------------------multi model prediction--------------------------
def multiPredict(K, Model, X, W):
    predict = []
    for i in range(K):
        pre = Model[i].predict(X)
        predict.append(pre)
    predict = mat(predict).T
    m, n = predict.shape
    # print(predict)
    ans = []
    for i in range(m):
        k = maxNum(predict[i], 0, 17)
        '''
		if maxNum(predict[i],6,12)==2:
			k=2
		elif maxNum(predict[i],0,3)==1:
			k=1
		elif maxNum(predict[i],6,12)==8:
			k=8
		elif maxNum(predict[i],0,3)==9:
			k=9
		elif maxNum(predict[i],6,12)==4:
			k=4
		elif maxNum(predict[i],12,15)==3:
			k=3
		'''
        ans.append(k)
    return mat(ans).T


def maxNum(X, a, b):
    num = [0] * 10
    for i in range(a, b):
        num[int(X[0, i])] += 1
    Max = 0
    k = 0
    for i in range(10):
        if num[i] > Max:
            Max = num[i]
            k = i
    return k


# --------------------------------main function-----------------------------
if __name__ == "__main__":
    # fx = 'data/test_feature.txt'
    # fy = 'data/test_label.txt'
    fx = 'challenge/cdigits_digits_vec.txt'
    fy = 'data/cdigits_digits_labels.txt'
    test_X = loadData(fx)
    test_Y = loadData(fy)

    Model = []
    # SVM 0-2
    for i in range(3):
        clf = joblib.load('model/parameters-SVM-' + str(i + 1) + '.pkl')
        Model.append(clf)
    # ANN 3-5
    for i in range(3):
        no = str(i + 1)
        clf = joblib.load('model/parameters-ANN-' + no + '.pkl')
        Model.append(clf)
    # KNN 6-11
    for i in range(5):
        no = str(i + 1)
        clf = joblib.load('model/parameters-KNN-' + no + '.pkl')
        Model.append(clf)
    # Random Forest 12-14
    for i in range(3):
        clf = joblib.load('model/parameters-RF-' + str(i + 1) + '.pkl')
        Model.append(clf)
    # Logistic Regression 15-17
    for i in range(3):
        clf = joblib.load('model/parameters-LR-' + str(i + 1) + '.pkl')
        Model.append(clf)

    weight = ones(17)
    predict = multiPredict(17, Model, test_X, weight)
    statistic(predict, test_Y)
