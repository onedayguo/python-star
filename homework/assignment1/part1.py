import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import scipy.io as scio
import matplotlib.pyplot as plt
import random

def get_data_from_file():
    # 1*50
    samp_x = np.loadtxt("D:/GitHub/python-star/PA-1-data-text/polydata_data_sampx.txt")
    # 50*1
    samp_y = np.loadtxt("D:/GitHub/python-star/PA-1-data-text/polydata_data_sampy.txt")
    # 1*100
    poly_x = np.loadtxt("D:/GitHub/python-star/PA-1-data-text/polydata_data_polyx.txt")
    # 100*1
    poly_y = np.loadtxt("D:/GitHub/python-star/PA-1-data-text/polydata_data_polyy.txt")
    return samp_x, samp_y, poly_x, poly_y


def get_data_from_matfile():
    data_path="E:/CS5487/L3/programming assigment/PA-1-data-matlab/poly_data.mat"
    data = scio.loadmat(data_path)
    sampx = data['sampx']  # (1,50)
    sampy = data['sampy']  # (50,1)
    polyx = data['polyx']  # (1,100)
    polyy = data['polyy']  # (100,1)
    return sampx, sampy, polyx, polyy


# generate K_th order polynominal feature -->(k,column)
def generate_k_order_data(dataSeed, k):
    kth = k + 1
    column = dataSeed.shape[1]
    dataSet = np.zeros((kth, column))
    for i in range(kth):
        for j in range(column):
            dataSet[i, j] = np.power(dataSeed[0, j], i)
    return dataSet


def ls_theta(samplex, sampley):  # output k*1
    return np.dot(np.matrix(np.dot(samplex, samplex.transpose())).I, samplex).dot(sampley)


def rls_theta(samplex, sampley, lamda):  # k*1
    theta = (np.matrix(np.dot(samplex, samplex.transpose()) +
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
    sigma_bar = np.matrix((1 / alpha) * np.identity(len(samplex)) + (1 / sigma2) * np.dot(samplex, samplex.transpose())).I
    miu_bar = (1 / sigma2) * sigma_bar.dot(samplex).dot(sampley)
    return miu_bar


def predict(data_x, theta):
    return np.dot(data_x.transpose(), theta)


# def br_predict(polyx, sigma_bar):
#     pre_covariance = np.dot(polyx.transpose(), covariance).dot(polyx)
#     pre_mean = np.dot(polyx.transpose(), mean)
#     return pre_covariance


# make a plot of samples and predict
def plot_data(title, sample_x, sample_y, polyx, polyy, predict_y):
    plt.figure()
    plt.scatter(sample_x.tolist(), sample_y.tolist(), color='black', label='samples',marker='.', linewidth=0.01)
    plt.scatter(polyx, polyy, color='blue', label='ploy', marker='*', linewidth=0.01)
    plt.plot(polyx.transpose(), predict_y, color='red', label='prediction', linewidth=1)
    plt.title(title)
    plt.xlabel('data_x')
    plt.ylabel('label_y')
    plt.grid(True)
    plt.legend()
    plt.show()


# make a plot standard deviation around the mean for BR
def plot_br_standard_deviation(title, sample_x, sample_y, polyx, polyy, predict_y, k):
    plt.figure()
    plt.scatter(sample_x.tolist(), sample_y.tolist(), color='black', label='samples', linewidth=0.1)
    plt.errorbar(polyx.T, predict_y, k, ecolor='red')
    plt.scatter(polyx, polyy, color='blue', label='ploy', linewidth=0.1)
    plt.title(title)
    plt.xlabel('data_x')
    plt.ylabel('label_y')
    plt.grid(True)
    plt.legend()
    plt.show()


# compute mean-square error(MSE)
def get_MSE(a, b):
    return np.square(a - b).mean()


# find the best hyperparameters which can reduce the MSE,  RLS（lamda）Lasso(lamda) BR(alpha, lamda)
def find_best_parameter(sigma2):
    global_lamda = np.arange(0, 1, 0.01)
    # get best parameters fro RLS
    rls_mse = 1.0
    rls_best_lamda = 0.0
    for lamda_i in global_lamda:
        rlsTheata = rls_theta(matrix_samplex, samp_y, lamda_i)
        rlsPredict = predict(matrix_poly_x, rlsTheata)
        mse = get_MSE(rlsPredict, poly_y)
        if mse < rls_mse:
            rls_mse = mse
            rls_best_lamda = lamda_i
    print('regularized LS lowest MSE is {0},best lamda is {1} '.format(rls_mse, rls_best_lamda))

    # get best parameters fro LASSO
    lasso_mse = 1.0
    lasso_best_lamda = 0.0
    for lamda_i in global_lamda:
        lassoTheata = lasso_theta(matrix_samplex, samp_y, kth_order, lamda_i)
        lassoPredict = predict(matrix_poly_x, lassoTheata)
        mse = get_MSE(lassoPredict, poly_y)
        if mse < lasso_mse:
            lasso_mse = mse
            lasso_best_lamda = lamda_i
    print('LASSO lowest MSE is {0},best lamda is {1} '.format(lasso_mse, lasso_best_lamda))

    # get best parameters fro Bayesian Regression(BR)
    alpha_set = np.arange(0.1, 100.0, 0.1)
    br_mse = 1.0
    br_best_alpha = 1.0
    for alpha_i in alpha_set:
        brTheata = br_theta(matrix_samplex, samp_y, alpha_i, sigma2=sigma2)
        brPredict = predict(matrix_poly_x, brTheata)
        mse = get_MSE(brPredict, poly_y)
        if mse < br_mse:
            br_mse = mse
            br_best_alpha = alpha_i
    print('Bayesian Regression lowest MSE is {0},best alpha is {1} '.format(br_mse, br_best_alpha))


# sampling num points randomly from samp_x, samp_y
def get_num_samples_randomly(sub_trains, all_trains, samp_x, samp_y):
    sample_index = sorted(random.sample(range(all_trains), sub_trains))
    data_x = []
    data_y = []
    for i in sample_index:
        data_x.append(samp_x[0, i])
        data_y.append(samp_y[i, 0])
    new_data_x = np.mat(data_x)
    new_data_y = np.transpose(np.mat(data_y))
    return new_data_x, new_data_y


def get_average_mse(sub_trains, all_trains, kth_order, samp_x, samp_y, poly_x, poly_y):
    mse = 0
    for i in range(100):
        data_x_i, data_y_i = get_num_samples_randomly(sub_trains, all_trains, samp_x, samp_y)
        mat_data_x_i = generate_k_order_data(data_x_i, kth_order)
        mat_poly_x = generate_k_order_data(poly_x, kth_order)
        # theta = ls_theta(mat_data_x_i, data_y_i) # 1 LS
        # theta = rls_theta(mat_data_x_i, data_y_i, rls_lamda) # 2 RLS
        # theta = lasso_theta(mat_data_x_i, data_y_i, kth_order, lasso_lamda) # 3 LASSO
        theta = rr_theta(mat_data_x_i, data_y_i, kth_order, sub_trains) # 4 RR
        # theta = br_theta(mat_data_x_i, data_y_i, br_alpha, sigma2) # 5 BR
        predict_y = predict(mat_poly_x, theta)
        mse_i = get_MSE(predict_y, poly_y)
        mse += mse_i
    return mse / 100


def plot_mse(sub_trains1, all_trains1, kth_order1, samp_x1, samp_y1, poly_x1, poly_y1):
    all_mse = []
    for i in range(sub_trains1, all_trains1):
        mse = get_average_mse(i, all_trains1, kth_order1, samp_x1, samp_y1, poly_x1, poly_y1)
        all_mse.append(mse)
    x_value = np.arange(sub_trains1, all_trains1, 1, dtype=int)
    plt.title('RR')
    plt.xlabel('x_samples')
    plt.ylabel('MSE')
    plt.plot(x_value, all_mse, 'red', label='predict', linewidth=2)
    plt.show()


def add_outliers(sampx, sampy, polyx, polyy):
    copy_sampy = np.copy(sampy)
    sample_index = sorted(random.sample(range(50), 5))
    for i in sample_index:
        copy_sampy[i] = sampy[i] + 150
    new_x = generate_k_order_data(sampx, kth_order)
    n_polyx = generate_k_order_data(polyx, kth_order)
    lspredicty = predict(n_polyx, ls_theta(new_x, copy_sampy))
    rlspredicty = predict(n_polyx, rls_theta(new_x, copy_sampy, rls_lamda))
    lassopredicty = predict(n_polyx, lasso_theta(new_x, copy_sampy, k=kth_order, lamda=lasso_lamda))
    rrpredicty = predict(n_polyx, rr_theta(new_x, copy_sampy, kth_order, sub_trains))
    brpredicty = predict(n_polyx, br_theta(new_x, copy_sampy, br_alpha, sigma2=sigma2))
    plot_data('LS', sampx, copy_sampy, polyx, polyy, lspredicty)
    plot_data('RLS', sampx, copy_sampy, polyx, polyy, rlspredicty)
    plot_data('LASSO', sampx, copy_sampy, polyx, polyy,lassopredicty)
    plot_data('RR', sampx, copy_sampy, polyx, polyy, rrpredicty)
    plot_data('BR', sampx, copy_sampy, polyx, polyy, brpredicty)


kth_order = 10
rls_lamda = 0.48
lasso_lamda = 0.0
sigma2 = 5.0
br_alpha = 10.4
all_trains = 50
sub_trains = 50
samp_x0, samp_y0, poly_x, poly_y = get_data_from_matfile()

#  10%,25%,50%,75%--> 6,12,25,36
samp_x, samp_y = get_num_samples_randomly(sub_trains=sub_trains, all_trains=all_trains, samp_x=samp_x0, samp_y=samp_y0)
matrix_samplex = generate_k_order_data(samp_x, kth_order)
matrix_poly_x = generate_k_order_data(poly_x, kth_order)
# 1.least-squares(LS)
lsTheta = ls_theta(matrix_samplex, samp_y)
lsPredict = predict(matrix_poly_x, lsTheta)
plot_data('least-squares(LS)', samp_x, samp_y, poly_x, poly_y, lsPredict)
# 2.regularized LS(RLS)
rlsTheata = rls_theta(matrix_samplex, samp_y, rls_lamda)
rlsPredict = predict(matrix_poly_x, rlsTheata)
plot_data('regularized LS(RLS)', samp_x, samp_y, poly_x, poly_y, rlsPredict)
# 3.L1-regularized LS(LASSO)
lassoTheta = lasso_theta(matrix_samplex, samp_y, kth_order, lasso_lamda)
lassoPredict = predict(matrix_poly_x, lassoTheta)
plot_data('L1-regularized LS(LASSO)', samp_x, samp_y, poly_x, poly_y, lassoPredict)
# 4.robust regression(RR)
rrTheta = rr_theta(matrix_samplex, samp_y, k=kth_order, sub_trains=sub_trains)
rrPredict = predict(matrix_poly_x, rrTheta)
plot_data('rebust regression(RR)', samp_x, samp_y, poly_x, poly_y, rrPredict)
# 5.Bayesian regression(BR)
brTheta = br_theta(matrix_samplex, samp_y, sigma2=sigma2, alpha=br_alpha)
brPredict = predict(matrix_poly_x, brTheta)
plot_data('Bayesian regression(BR)', samp_x, samp_y, poly_x, poly_y, brPredict)
# 6.standard deviation around the mean for BR
plot_br_standard_deviation('standard deviation around mean', samp_x, samp_y, poly_x, poly_y, brPredict, kth_order)
# 7.compute mean-square error(MSE)
print('least-squares MSE = ', get_MSE(lsPredict, poly_y))
print('regularized LS MSE = ', get_MSE(rlsPredict, poly_y))
print('L1-regularized LS MSE = ', get_MSE(lassoPredict, poly_y))
print('robust regression MSE = ', get_MSE(rrPredict, poly_y))
print('bayesian regression MSE = ', get_MSE(brPredict, poly_y))

# 8.find best hyperparameters for RLS(lamda),LASSO(lamda),BR(lamda,alpha) find_best_parameter()
# find_best_parameter(sigma2=sigma2)
# 9.sample 10%,25%,50%,75% and plot estimate functions from 50 samples,but to reuse code, i need put
# get_num_samples_randomly() before. so you can find the function under the function get_data_from_matfile()
# get_num_samples_randomly(num=5)  # 6,12,25,36
# 10.make a plot of average MSE
# plot_mse(sub_trains, all_trains, kth_order, samp_x0, samp_y0, poly_x, poly_y)
# 11.add outliers to few samples and plot again
# add_outliers(samp_x0, samp_y0, poly_x, poly_y)