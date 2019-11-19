import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import scipy.io as scio
import matplotlib.pyplot as plt
import random


def get_data_from_matfile():
    data_path="E:/CS5487/L3/programming assigment/PA-1-data-matlab/count_data.mat"
    data = scio.loadmat(data_path)
    trainx = data['trainx']  # 9*400
    trainy = data['trainy']  # 400*1
    testx = data['testx']    # 9*600
    testy = data['testy']    # 600*1
    return trainx, trainy, testx, testy


# generate K_th order polynominal feature -->(k,column)
def generate_k_order_data(dataSeed, k):
    kth = k + 1
    column = dataSeed.shape[1]
    dataSet = np.zeros((kth, column))
    for i in range(kth):
        for j in range(column):
            dataSet[i, j] = np.power(dataSeed[0, j], i)
    return dataSet


def power_k(dataSet, k):  # k is order
    newDataSet = np.zeros(dataSet.shape)  # 9*400
    clumn = dataSet.shape[1]  # 400
    row = dataSet.shape[0]  # 9
    for i in range(clumn):
        for j in range(row):
            newDataSet[j, i] = np.power(dataSet[j, i], k)
    # syn_xx=np.vstack((datatype, xx))
    return newDataSet


def ls_theta(samplex, sampley):  # output k*1
    return np.dot(np.mat(np.dot(samplex, samplex.transpose())).I, samplex).dot(sampley)


def rls_theta(samplex, sampley, lamda):  # k*1
    theta = (np.mat(np.dot(samplex, samplex.transpose()) +
                       lamda * np.identity(len(samplex))).I).dot(samplex).dot(sampley)
    return theta


def lasso_theta(samplex, sampley, k, lamda):
    first = np.dot(samplex, samplex.transpose())
    second = np.dot(samplex, sampley)
    third = np.concatenate((second, -1 * second), axis=0)
    Hl = np.concatenate((first, -1 * first), axis=0)
    Hr = np.concatenate((-1 * first, first), axis=0)
    H = np.concatenate((Hl, Hr), axis=1)
    f = lamda * np.ones((len(third), 1)) - third
    G = -1 * np.identity((len(H)))
    value = np.zeros((len(H), 1))
    t = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(value))['x']
    theta = np.mat([t[i] - t[i + k+1] for i in range(int(len(t) / 2))]).transpose()
    return theta


def rr_theta(samplex, sampley, k, sub_trains):
    f1 = np.concatenate((np.zeros((k+1, 1)), np.ones((sub_trains, 1))), axis=0)
    Al = np.concatenate((-1 * samplex.transpose(), samplex.transpose()), axis=0)
    Ar = np.concatenate((-1 * np.identity(sub_trains), -1 * np.identity(sub_trains)), axis=0)
    A = np.concatenate((Al, Ar), axis=1)
    b = np.concatenate((-1 * sampley, sampley), axis=0)
    return solvers.lp(matrix(f1), matrix(A), matrix(b))['x'][0:k+1]


def br_theta(samplex, sampley, alpha, sigma2):
    sigma_bar = np.mat((1 / alpha) * np.identity(len(samplex)) + (1 / sigma2) * np.dot(samplex, samplex.transpose())).I
    miu_bar = (1 / sigma2) * sigma_bar.dot(samplex).dot(sampley)
    return miu_bar


def predict(data_x, theta):
    return np.dot(data_x.transpose(), theta)


# def br_predict(polyx, sigma_bar):
#     pre_covariance = np.dot(polyx.transpose(), covariance).dot(polyx)
#     pre_mean = np.dot(polyx.transpose(), mean)
#     return pre_covariance


# make a plot of samples and predict
def plot_data(title, x, y, predict_y):
    plt.figure()
    plt.title(title)
    # plt.scatter(sample_x, sample_y,color='black',label='samples',linewidth=0.1) # sample data
    plt.plot(x, predict_y, 'r', label='predict', linewidth=2)  # test data
    plt.plot(x, y, 'b', label='label', linewidth=2)
    # plt.scatter(x,y,color='blue',label='ploy',linewidth=0.1)
    # plt.errorbar(x.T,predict_y(polyx,BR(newx, sampy,5)),5,fmt='.k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_predict(trainx, trainy, testx, testy, lamda0, kth, subTrains, alpha0, sigma02):
    lsTheta = ls_theta(trainx, trainy)
    lsPredict = np.round(predict(testx, lsTheta))
    rlsTheta = rls_theta(trainx, trainy, lamda0)
    rlsPredict = np.round(predict(testx, rlsTheta))
    lassoTheta = lasso_theta(trainx, trainy, kth, lamda0)
    lassoPredict = np.round(predict(testx, lassoTheta))
    rrTheta = rr_theta(trainx, trainy, kth, subTrains)
    rrPredict = np.round(predict(testx, rrTheta))
    brTheta = br_theta(trainx, trainy, alpha0, sigma02)
    brPredict = np.round(predict(testx, brTheta))
    x_value = np.linspace(1, 600, 600)

    plot_data("LS",x_value, test_y, lsPredict)
    plot_data("RLS", x_value, test_y, rlsPredict)
    plot_data("LASSO", x_value, test_y, lassoPredict)
    plot_data("RR", x_value, test_y, rrPredict)
    plot_data("BR", x_value, test_y, brPredict)


# compute mean-square error(MSE)
def get_MSE(a, b):
    return np.square(a - b).mean()


# compute mean-ablolute error(MAE)
def get_MAE(a, b):
    return np.abs(a - b).mean()


# print errors including MSE,MAE of test data
def print_error(trainx, trainy, testx, testy, lamda0, kth, subTrains, alpha0, sigma02):
    lsTheta = ls_theta(trainx, trainy)
    lsPredict = predict(testx, lsTheta)
    print('LS MSE', get_MSE(lsPredict, testy))
    print('LS MAE', get_MAE(lsPredict, testy))

    rlsTheta = rls_theta(trainx, trainy, lamda0)
    rlsPredict = predict(testx, rlsTheta)
    print('RLS MSE', get_MSE(rlsPredict, testy))
    print('RLS MAE', get_MAE(rlsPredict, testy))

    lassoTheta = lasso_theta(trainx, trainy, kth, lamda0)
    lassoPredict = predict(testx, lassoTheta)
    print('LASSO MSE', get_MSE(lassoPredict, testy))
    print('LASSO MAE', get_MAE(lassoPredict, testy))

    rrTheta = rr_theta(trainx, trainy, kth, subTrains)
    rrPredict = predict(testx, rrTheta)
    print('RR MSE', get_MSE(rrPredict, testy))
    print('RR MAE', get_MAE(rrPredict, testy))

    brTheta = br_theta(trainx, trainy, alpha0, sigma02)
    brPredict = predict(testx, brTheta)
    print('BR MSE',get_MSE(brPredict, testy))
    print('RR MAE', get_MAE(brPredict, testy))


def cross_data(data):
    new_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        for j in range(9):
            if j < 8:
                new_data[j, i] = data[j, i] * data[j+1, i]
            else:
                new_data[j, i] = data[j, i] * data[0, i]
    return new_data


def problemB(trainx, trainy,testx, testy):
    # 2nd order polynomial
    sample_x = np.vstack((trainx, power_k(trainx, 2)))
    sample_y = trainy
    test_x = np.vstack((testx, power_k(testx, 2)))
    test_y =testy

    print('2nd order least-squares MSE', get_MSE(predict(test_x, ls_theta(sample_x, sample_y)), test_y))
    print('2nd order regularized LS MSE', get_MSE(predict(test_x, rls_theta(sample_x, sample_y, lamda)), test_y))
    print('2nd order L1-regularized LS MSE', get_MSE(predict(test_x, lasso_theta(sample_x, sample_y, 8, lamda)), test_y))
    print('2nd order robust regression MSE', get_MSE(predict(test_x, rr_theta(sample_x, sample_y, 8, 400)), test_y))
    print('2nd bayesian regression MSE', get_MSE(predict(test_x, br_theta(sample_x, sample_y, br_alpha, sigma2)), test_y))
    print('-------------------------------------------------')
    print('2nd order least-squares MAE', get_MAE(predict(test_x, ls_theta(sample_x, sample_y)), test_y))
    print('2nd order regularized LS MAE', get_MAE(predict(test_x, rls_theta(sample_x, sample_y, lamda)), test_y))
    print('2nd order L1-regularized LS MAE', get_MAE(predict(test_x, lasso_theta(sample_x, sample_y, 8, lamda)), test_y))
    print('2nd order robust regression MAE', get_MAE(predict(test_x, rr_theta(sample_x, sample_y, 8, 400)), test_y))
    print('2nd bayesian regression MAE', get_MAE(predict(test_x, br_theta(sample_x, sample_y, br_alpha, sigma2)), test_y))
    # 3rd order polynomial
    sample_x3 = np.vstack((trainx, power_k(trainx, 3)))
    test_x3 = np.vstack((testx, power_k(testx, 3)))
    print('-------------------------------------------------')
    print('3nd order least-squares MSE', get_MSE(predict(test_x3, ls_theta(sample_x3, sample_y)), test_y))
    print('3nd order regularized LS MSE', get_MSE(predict(test_x3, rls_theta(sample_x3, sample_y, lamda)), test_y))
    print('3nd order L1-regularized LS MSE', get_MSE(predict(test_x3, lasso_theta(sample_x3, sample_y, 5,lamda)), test_y))
    print('3nd order robust regression MSE', get_MSE(predict(test_x3, rr_theta(sample_x3, sample_y, 5, 400)), test_y))
    print('3nd bayesian regression MSE', get_MSE(predict(test_x3, br_theta(sample_x3, sample_y, 5)), test_y))

    print('3nd order least-squares MAE', get_MAE(predict(test_x3, ls_theta(sample_x3, sample_y)), test_y))
    print('3nd order regularized LS MAE', get_MAE(predict(test_x3, rls_theta(sample_x3, sample_y, lamda)), test_y))
    print('3nd order L1-regularized LS MAE', get_MAE(predict(test_x3, lasso_theta(sample_x3, sample_y, 5)), test_y))
    print('3nd order robust regression MAE', get_MAE(predict(test_x3, rr_theta(sample_x3, sample_y, 5, 400)), test_y))
    print('3nd bayesian regression MAE', get_MAE(predict(test_x3, br_theta(sample_x3, sample_y, br_alpha, sigma2)), test_y))


def problemBcross(trainx, trainy, testx, testy):
    sample_x = np.vstack((trainx, cross_data(trainx)))
    sample_y = trainy
    test_x = np.vstack((testx, cross_data(testy)))
    test_y = testy
    print('cross order least-squares MSE', get_MSE(predict(test_x, ls_theta(sample_x, sample_y)), test_y))
    print('crossorder regularized LS MSE', get_MSE(predict(test_x, rls_theta(sample_x, sample_y, lamda)), test_y))
    print('cross order L1-regularized LS MSE', get_MSE(predict(test_x, lasso_theta(sample_x, sample_y, 10)), test_y))
    print('cross order robust regression MSE', get_MSE(predict(test_x, rr_theta(sample_x, sample_y, k, 400)), test_y))
    print('cross bayesian regression MSE', get_MSE(predict(test_x, br_theta(sample_x, sample_y, br_alpha, sigma2)), test_y))
    print('-------------------------------------------------')
    print('cross order least-squares MAE', get_MAE(predict(test_x, lasso_theta(sample_x, sample_y)), test_y))
    print('cross order regularized LS MAE', get_MAE(predict(test_x, rls_theta(sample_x, sample_y, lamda)), test_y))
    print('cross order L1-regularized LS MAE', get_MAE(predict(test_x, lasso_theta(sample_x, sample_y, 10)), test_y))
    print('cross order robust regression MAE', get_MAE(predict(test_x, rr_theta(sample_x, sample_y, 10, 400)), test_y))
    print('cross bayesian regression MAE', get_MAE(predict(test_x, br_theta(sample_x, sample_y, br_alpha, sigma2)), test_y))

kth_order = 10
rls_lamda = 0.5
lasso_lamda = 0.0
lamda = 0.5
k = 8
n = 400
sigma2 = 5.0
br_alpha = 5.0

# 1.1get dataset from local file, print the error
train_x, train_y, test_x, test_y = get_data_from_matfile()
# print_error(train_x, train_y, test_x, test_y, lamda, k, n, br_alpha, sigma2)
# 1.2 plot the test and predict
# plot_predict(train_x, train_y, test_x,test_y,lamda,k,n,br_alpha,sigma2)
# 2.1 get a simple 2nd order
# problemB(train_x, train_y, test_x, test_y)