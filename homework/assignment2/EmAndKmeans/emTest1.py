# coding:gbk
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


def ini_data(sigma, init_mu, mix, N):
    data = np.zeros((1, N))                # 1行N列的的矩阵，型如【0,0,0,0,0,0，...】
    for i in range(N):
        temp = np.random.random(1)
        if temp < mix[0]:
            data[0, i] = np.random.normal() * sigma[0] + init_mu[0]
        else:
            data[0, i] = np.random.normal() * sigma[1] + init_mu[1]
    return data


def exe_em(sigma, init_mu, mix, N, iter, diff, pre_expec0, pre_mix0, pre_sigma0, pre_mu0):

    data = ini_data(sigma, init_mu, mix, N)

    K = len(mix)              # 混合系数个数，即分类个数
    pre_expec = pre_expec0    # 进行迭代期望矩阵
    pre_mix = pre_mix0        # 进行迭代的混合系数
    pre_sigma = pre_sigma0    # 进行迭代的方差
    pre_mu = pre_mu0          # 进行迭代的均值

    # EM迭代

    for it in range(iter):
        old_Mu = copy.deepcopy(pre_mu)

        # e step，计算期望矩阵
        for i in range(N):
            down = 0
            for j in range(K):
                # 用于计算期望矩阵的分母部分
                d_u = float(data[0, i]-pre_mu[j])
                down += math.exp(-1 / 2.0 * math.pow(d_u / 3.0, 2))
            for j in range(K):
                # 用于计算期望矩阵的分子部分
                d_u = float(data[0, i] - pre_mu[j])
                up = math.exp(-1 / 2.0 * math.pow(d_u / 3.0, 2))
                pre_expec[i, j] = up / down   # 更新第i条数据属于第j个模型的概率，即数据的概率密度函数

        # m step
        for j in range(K):
            up = 0
            down = 0
            sigma_up = 0
            for i in range(N):
                up += pre_expec[i, j] * data[0, i]
                down += pre_expec[i, j]
                sigma_up += pre_expec[i, j] * (data[0, i] - pre_mu[j])
            pre_mu[j] = up / down
            pre_mix[j] = down / N
            pre_sigma[j] = sigma_up / down

        print(it, "预测均值 ", pre_mu)
        print("预测方差 ", pre_sigma)
        print("预测混合系数 ", pre_mix)
        # if  < diff:
        #    break


if __name__ == '__main__':
    sigma = [3.0, 3.0]  # 初始化数据方差，一维数据
    init_mu = [25, 65]  # 初始化数据均值
    mix = [0.6, 0.4]  # 混合系数
    K = 2           # 数据种类，需要和方差均值数量对应
    N = 10000       # 数据数量总和
    iter = 100      # 最大迭代次数
    diff = 0.001    # 均值方差与上轮差异阈值

    pre_expec0 = np.zeros([N, K])  # 声明期望数组，意义：第N条数据属于第K个模型的概率
    pre_expec0 = np.mat(pre_expec0)
    pre_mix0 = [0.5, 0.5]          # 预测模型混合系数初始化，意义：每个模型下的数据所占比例
    pre_sigma0 = [5.0, 5.0]        # 预测方差初始化 ，意义：每个模型的方差
    pre_mu0 = np.array([10.0, 10.0])        # 预测均值初始化 ，意义：每个模型的均值

    exe_em(sigma, init_mu, mix, N, iter, diff, pre_expec0,pre_mix0, pre_sigma0, pre_mu0)

    # print(alpha)
    # plt.hist(data[0, :], 100)
    # plt.grid(True)
    # plt.show()