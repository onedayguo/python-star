from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.externals import joblib
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
def multiPredict(K, Model, X):
    predict = []
    for i in range(K):
        pre = Model[i].predict(X)
        predict.append(pre)
    predict = mat(predict).T
    m, n = predict.shape
    # print(predict)
    ans = []
    for i in range(m):
        num = [0] * 10
        for j in range(n):
            num[int(predict[i, j])] += 1
        Max = 0
        k = 0
        for j in range(10):
            if num[j] > Max:
                Max = num[j]
                k = j
        ans.append(k)
    return mat(ans).T


# --------------------------------main function-----------------------------
if __name__ == "__main__":
    fx = 'data/train_feature.txt'
    fy = 'data/train_label.txt'
    train_X = loadData(fx)
    train_Y = loadData(fy)

    fx = 'data/test_feature.txt'
    fy = 'data/test_label.txt'
    test_X = loadData(fx)
    test_Y = loadData(fy)

    '''
	clf=LogisticRegression(max_iter=5)
	clf.fit(train_X,train_Y)
	print(clf.score(train_X,train_Y))
	print(clf.score(test_X,test_Y))
	
	'''
    K = 3
    Model = []
    kf = KFold(n_splits=K)
    for train, test in kf.split(train_X):
        clf = LogisticRegression(max_iter=5)
        clf.fit(train_X[train], train_Y[train])
        score = clf.score(train_X[test], train_Y[test])
        Model.append(clf)
    for i in range(K):
        joblib.dump(Model[i], 'model/parameters-LR-' + str(i + 1) + '.pkl')
    predict = multiPredict(K, Model, train_X)
    statistic(predict, train_Y)
    predict = multiPredict(K, Model, test_X)
    statistic(predict, test_Y)
