"""
Created on Mon Oct 2 10:40:52 2019
@author: guozhenhui
"""
import random
import numpy as np
import cvxopt as opt
import cvxpy as cvx
import matplotlib.pyplot as plt
import scipy.io as scio

# <editor-fold desc="init global data">
data_path = "E:/CS5487/L3/programming assigment/PA-1-data-matlab/poly_data.mat"
data = scio.loadmat(data_path)
np.set_printoptions(suppress=True)
# lumda = 0.5
# sigma_o = 5
# percent = 100
sampx = data['sampx']  # (1,50)
sampy = data['sampy']  # (50,1)
polyx = data['polyx']  # (1,100)
polyy = data['polyy']  # (100,1)
# </editor-fold>


# generate K_th order polynominal feature -->(k,column)
def generate_k_order_data(k, dataSeed):
    kth = k + 1
    column = dataSeed.shape[1]
    dataSet = np.zeros((kth, column))
    for i in range(kth):
        for j in range(column):
            dataSet[i, j] = np.power(dataSeed[0, j], i)
    return dataSet


# sampling num points randomly from sampx, sampy
def sampling(num):
    sample_index = sorted(random.sample(range(50), num))
    x = []
    y = []
    for i in sample_index:
        x.append(sampx[0, i])
        y.append(sampy[i, 0])
    new_sampx = np.matrix(x)
    new_sampy = np.transpose(np.matrix(y))
    new_x = generate_k_order_data(6, new_sampx)
    return new_x, new_sampx, new_sampy


# transfer the array to list
def ArrayToList(array):
    list = []
    for i in range(len(array)):
        list.append(float(array[i]))
    return list


# LS regression
def ls_theta(data_x, label_y):
    theta = np.dot(np.matrix(np.dot(data_x, data_x.transpose())).I, data_x).dot(label_y)
    # ls = np.linalg.inv(xx @ xx.T) @ xx @ y
    return theta


# RLS regression
def rls_theta(data_x, label_y, lamda):
    theta = (np.matrix(np.dot(data_x, data_x.transpose()) +
                       lamda * np.identity(len(data_x))).I).dot(data_x).dot(label_y)
    # rls = np.linalg.inv(data_x @ data_x.T + lamda) @ data_x @ label_y
    return theta


# Lasso regression
def lasso_theta(k, data_x, label_y, lamda):
    p_temp = np.vstack((np.hstack([data_x @ np.transpose(data_x), -1 * data_x @ np.transpose(data_x)]),
                        np.hstack([-1 * data_x @ np.transpose(data_x), data_x @ np.transpose(data_x)])))
    f_temp = lamda * np.ones((2 * k + 2, 1)) - np.vstack([data_x @ label_y, -1 * data_x @ label_y])
    G_temp = -1 * np.eye(2 * k + 2)
    h_temp = np.zeros((2 * k + 2, 1))
    P = opt.matrix(p_temp)
    f = opt.matrix(f_temp)
    G = opt.matrix(G_temp)
    h = opt.matrix(h_temp)
    x_par = np.array(opt.solvers.qp(P, f, G, h)['x'])
    thetaPlus = x_par[: k + 1, :]
    thetaMinus = x_par[k + 1:, :]
    return thetaPlus - thetaMinus


# RR regression
def rr_theta(xx, y, k, n):
    c = np.vstack((np.zeros((k + 1, 1)), np.ones((n, 1))))
    b = np.vstack((-1 * y, y))
    A = np.vstack((np.hstack((-1 * np.transpose(xx), -1 * np.eye(n))), np.hstack((np.transpose(xx), -1 * np.eye(n)))))
    B = ArrayToList(b)

    x = cvx.Variable(n + k + 1)
    obj = cvx.Minimize(c.T * x)
    constraints = [A * x <= B]
    prob = cvx.Problem(obj, constraints)
    res = prob.solve()
    theta = x.value[: k + 1]
    return np.reshape(theta, (-1, 1))


# bayesian regression
def br_theta(data_x, y, alpha, sigma):
    sigma_par = np.linalg.inv(1 / alpha + 1 / sigma * data_x @ data_x.T)
    miu_par = 1 / sigma * sigma_par @ data_x @ y
    return miu_par


# MSE
def get_MSE(a, b):
    return np.square(a - b).mean()


# predict the Y value of x using theta
def predict_y(x, theta):
    predict_y = x.T @ theta
    return predict_y


# plot the data and the line
def plot_data(title, sample_x, sample_y, test_x, predict_y, test_y):
    plt.figure()
    plt.title(title)
    # plt.scatter(sample_x, sample_y, color='black', label='samples', linewidth=0.1)  # sample data
    plt.plot(test_x.T, predict_y, 'r', label='predict', linewidth=2)  # test data
    plt.scatter(test_x, test_y, color='y', label='ploy', linewidth=0.1)
    # plt.errorbar(x.T,predict_y(polyx,BR(newx, sampy,5)),5,fmt='.k')
    plt.legend()


# use iteration to get the theta which can work out lowest MSE
def optimum_parameter():
    newx = generate_k_order_data(6, sampx)
    n_polyx = generate_k_order_data(6, polyx)
    a2 = np.arange(0, 200, 0.05)
    c = 0
    min = 1
    for a in a2:
        x = get_MSE(predict_y(n_polyx, br_theta(newx, sampy, a)), polyy)
        if x < min:
            min = x
            c = a
            if x > min:
                min = min
    print(min, c)


def plot_BR(sample_x, sample_y, x, predcit_y, y, var):
    plt.scatter(sample_x, sample_y, color='black', label='samples', linewidth=0.1)  # sample data
    plt.errorbar(x, predcit_y, np.sqrt(var), color='red', lable='predict')  # test data
    plt.scatter(x, y, color='blue', label='ploy', linewidth=0.1)
    plt.legend()


# plt.errorbar(x, y, dy, fmt='.k', ecolor='lightgray', elinewidth=3 capsize=0)
# plt.errorbar(poly_x, predict_y(po_x,BR(xx,5)),5,fmt='.k')
# plt.plot(poly_x.T,predict_y(po_x,BR(xx,y,5)))
# questionb()
# compute MSE after sampling
def sampling_test(i):
    sample_x = sampling(i)[0]
    sample_y = sampling(i)[2]

    n_polyx = generate_k_order_data(6, polyx)
    lsMSE = get_MSE(predict_y(n_polyx, ls_theta(sample_x, sample_y)), polyy)
    return lsMSE


#    rlsMSE=MSE(predict_y(n_polyx,rls(newx, sampy,0.5)),polyy)
#    lassoMSE=MSE(predict_y(n_polyx,lasso(newx, sampy)),polyy)
#    rrMSE=MSE(predict_y(n_polyx,rr(newx, sampy,50)),polyy)
#    BRMSE=MSE(predict_y(n_polyx,BR(newx, sampy,5)),polyy)
# ----------计算MSE平均值
# ssum=0
# for i in range(500):
#    ssum+=samplingtest(20)
# print(ssum/500)

# -------------------------------
def sampling_plot(i):
    new_x = np.array(sampling(i)[0])
    sample_x = np.array(sampling(i)[1])
    sample_y = np.array(sampling(i)[2])
    n_polyx = generate_k_order_data(6, polyx)
    lspredicty = predict_y(n_polyx, ls_theta(new_x, sample_y))
    rlspredicty = predict_y(n_polyx, rls_theta(new_x, sample_y, 0.095))
    lassopredicty = predict_y(n_polyx, lasso_theta(new_x, sample_y, 0.0))
    rrpredicty = predict_y(n_polyx, rr_theta(new_x, sample_y, i))
    brpredicty = predict_y(n_polyx, br_theta(new_x, sample_y, 52.15))
    # rint( predict_y(n_polyx,ls(new_x, sample_y)) )    # prediction of , ls()genereate theta
    plot_data('Ls', sample_x, sample_y, polyx, lspredicty, polyy)
    plot_data('Rls', sample_x, sample_y, polyx, rlspredicty, polyy)
    plot_data('Lasso', sample_x, sample_y, polyx, lassopredicty, polyy)
    plot_data('RR', sample_x, sample_y, polyx, rrpredicty, polyy)
    plot_data('BR', sample_x, sample_y, polyx, brpredicty, polyy)
    # plt.scatter(sample_x, sample_y.T


# samplingplot(38)


def questionb():
    n_polyx = generate_k_order_data(6, polyx)
    newx = generate_k_order_data(6, sampx)
    print('least-squares MSE', get_MSE(predict_y(n_polyx, ls_theta(newx, sampy)), polyy))
    print('regularized LS MSE', get_MSE(predict_y(n_polyx, rls_theta(newx, sampy, 0.5)), polyy))
    print('L1-regularized LS MSE', get_MSE(predict_y(n_polyx, lasso_theta(newx, sampy)), polyy))
    print('robust regression MSE', get_MSE(predict_y(n_polyx, rr_theta(newx, sampy, 50)), polyy))
    print('bayesian regression MSE', get_MSE(predict_y(n_polyx, br_theta(newx, sampy, 5)), polyy))


def diffrentsizeMSEplot():
    lsmse = np.zeros(50)
    rlsmse = np.zeros(50)
    lassomse = np.zeros(50)
    rrmse = np.zeros(50)
    brmse = np.zeros(50)
    plt.figure()
    for a in range(1, 50):
        new_x = np.array(sampling(a)[0])
        # sample_x=np.array(sampling(a)[1])
        sample_y = np.array(sampling(a)[2])
        n_polyx = generate_k_order_data(6, polyx)
        lspredicty = predict_y(n_polyx, ls_theta(new_x, sample_y))
        rlspredicty = predict_y(n_polyx, rls_theta(new_x, sample_y, 0.095))
        lassopredicty = predict_y(n_polyx, lasso_theta(new_x, sample_y, 0.0))
        rrpredicty = predict_y(n_polyx, rr_theta(new_x, sample_y, a))
        brpredicty = predict_y(n_polyx, br_theta(new_x, sample_y, 52.15))

        lsmse[a] = get_MSE(polyy, lspredicty)
        rlsmse[a] = get_MSE(rlspredicty, polyy)
        lassomse[a] = get_MSE(lassopredicty, polyy)
        rrmse[a] = get_MSE(rrpredicty, polyy)
        brmse[a] = get_MSE(brpredicty, polyy)
    print(lsmse)
    # plt.plot(a,lsmse[a])


#    Plotdata('Ls',sample_x, sample_y, polyx,lspredicty,polyy)
#    Plotdata('Rls',sample_x, sample_y, polyx, rlspredicty,polyy)
#    Plotdata('Lasso',sample_x, sample_y, polyx, lassopredicty,polyy)
#    Plotdata('RR',sample_x, sample_y, polyx, rrpredicty,polyy)
#    Plotdata('BR',sample_x, sample_y, polyx, brpredicty,polyy)

# diffrentsizeMSEplot()
# find the m
def findallmseofls():
    lsmse = np.zeros(45)
    for i in range(5, 50):
        new_x = np.array(sampling(i)[0])
        # sample_x=np.array(sampling(a)[1])
        sample_y = np.array(sampling(i)[2])
        n_polyx = generate_k_order_data(6, polyx)
        lspredicty = predict_y(n_polyx, ls_theta(new_x, sample_y))
        lsmse[i] = get_MSE(lspredicty, polyy)
    print(lsmse)


def average_mse(a):
    summ = 0
    for i in range(100):
        new_x = np.array(sampling(a)[0])
        sample_y = np.array(sampling(a)[2])
        n_polyx = generate_k_order_data(6,polyx)
        lspredicty = predict_y(n_polyx, lasso_theta(new_x, sample_y, 0.1))
        lsmse = get_MSE(lspredicty, polyy)
        summ += lsmse
    return summ / 100


def plot_mse():
    lsmse = []
    for i in range(8, 50):
        lsmse.append(average_mse(i))
    MSEls = np.array(lsmse)
    x1 = np.linspace(8, 50, 42)
    plt.title('Lasso_MSE')
    plt.plot(x1, MSEls, 'r', label='predict', linewidth=2)


def addsampy():
    addsampy = np.copy(sampy)
    sample_index = sorted(random.sample(range(50), 4))
    for i in sample_index:
        addsampy[i] = sampy[i] + 120
    new_x = generate_k_order_data(6, sampx)
    n_polyx = generate_k_order_data(6, polyx)
    lspredicty = predict_y(n_polyx, ls_theta(new_x, addsampy))
    rlspredicty = predict_y(n_polyx, rls_theta(new_x, addsampy, 0.095))
    lassopredicty = predict_y(n_polyx, lasso_theta(new_x, addsampy, 0.5))
    rrpredicty = predict_y(n_polyx, rr_theta(new_x, addsampy, 50))
    brpredicty = predict_y(n_polyx, br_theta(new_x, addsampy, 52.15))
    plot_data('Ls', sampx, addsampy, polyx, lspredicty, polyy)
    plot_data('Rls', sampx, addsampy, polyx, rlspredicty, polyy)
    plot_data('Lasso', sampx, addsampy, polyx, lassopredicty, polyy)
    plot_data('RR', sampx, addsampy, polyx, rrpredicty, polyy)
    plot_data('BR', sampx, addsampy, polyx, brpredicty, polyy)


# new_x=np.array(sampling(6)[0])
# sample_y=np.array(sampling(6)[2])
# n_polyx=generatedata(polyx)
# lspredicty=predict_y(n_polyx,ls(new_x, sample_y))
# lsmse=MSE(lspredicty,polyy)
# print(lsmse)


kth_order = 5
newx = generate_k_order_data(kth_order, sampx)
n_polyx = generate_k_order_data(kth_order, polyx)
ls_pre_y = predict_y(n_polyx, ls_theta(newx, sampy))
rls_pre_y = predict_y(n_polyx, rls_theta(newx, sampy, 0.5))
lasso_pre_y = predict_y(n_polyx, lasso_theta(10, newx, sampy, 0.5))
rr_pre_y = predict_y(n_polyx, rr_theta(newx, sampy, 10, 50))
br_pre_y = predict_y(n_polyx, br_theta(newx, sampy, 52.15))
plot_data('LS', sampx, sampy, polyx, ls_pre_y, polyy)
plot_data('RLS', sampx, sampy, polyx, rls_pre_y, polyy)
plot_data('LASSO', sampx, sampy, polyx, lasso_pre_y, polyy)
plot_data('RR', sampx, sampy, polyx, rr_pre_y, polyy)
# plot_data('RR',sampx, addsampy, polyx, rrpredicty,polyy)
#plot_data('RR', sampx, sampy, polyx, brpredicty, polyy)
plot_data('BR', sampx, sampy, polyx, br_pre_y, polyy)
plt.show()
print('least-squares MSE', get_MSE(predict_y(n_polyx, ls_theta(newx, sampy)), polyy))
print('regularized LS MSE', get_MSE(predict_y(n_polyx, rls_theta(newx, sampy, 0.095)), polyy))
print('L1-regularized LS MSE', get_MSE(predict_y(n_polyx, lasso_theta(10, newx, sampy, 0.0)), polyy))
print('robust regression MSE',get_MSE(predict_y(n_polyx,rr_theta(10, newx, sampy, 50)), polyy))
# print(rr_theta(newx, sampy, 10, 50))
print('bayesian regression MSE', get_MSE(predict_y(n_polyx, br_theta(newx, sampy, 52.15)), polyy))

newx=generatedata(sampx)
n_polyx=generatedata(polyx)            # parameters
predict_y = predict_y(n_polyx, BR(newx, sampy,5))     # prediction of , ls()genereate theta
#Plotdata('RR',sampx, sampy, polyx, predict_y,polyy)
#plotBR(sampx, sampy, polyx ,predict_y,polyy, 5)
plt.scatter(sampx, sampy,color='black',label='samples',linewidth=0.1)
# sample data
plt.errorbar(polyx, predict_y,5 )# test data
plt.scatter(polyx, polyy,color='blue',label='ploy',linewidth=0.1)
plt.legend()
