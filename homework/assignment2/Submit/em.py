# coding:gbk
from numpy import *
import math
import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

def ini_data(sigma,init_mu,mix,K,N):
    data = np.zeros((1, N))                # 1行N列的的矩阵，型如【0,0,0,0,0,0，...】
    for i in range(N):
        temp = np.random.random(1)
        if temp < mix[0]:
            data[0, i] = np.random.normal() * sigma[0] + init_mu[0]
        else:
            data[0, i] = np.random.normal() * sigma[1] + init_mu[1]
    return data

# 算法迭代iter_num次，或达到精度Epsilon停止迭代
def exe_em(Sigma, iter_num, diff,data):


    K = 2
    N = len(data)
    pre_exp = np.zeros((N, K))  # N行K列的矩阵，记录异常，这里k是2，N行2列
    pre_mix =np.array([0.4, 0.6])
    pre_mu = np.random.random(2)  # 预测均值初始化 ，意义：每个模型的均值
    for it in range(iter_num):

        # e step
        last_Mu = copy.deepcopy(pre_mu)
        last_mix = copy.deepcopy(pre_mix)
        for i in range(0, N):
            down = 0
            for j in range(0, K):
                d_u = (data[i] - pre_mu[j])
                down += math.exp(-1 / 2.0 * math.pow(d_u / Sigma, 2))
            for j in range(0, K):
                d_u = (data[i] - pre_mu[j])
                up = math.exp(-1 / 2.0 * math.pow(d_u / Sigma, 2))
                pre_exp[i, j] = up / down
        # m step
        for j in range(K):
            up = 0
            down = 0
            for i in range(0, N):
                up += pre_exp[i, j] * data[i]
                down += pre_exp[i, j]
            pre_mu[j] = up / down
            pre_mix[j] = down / N
        print("迭代次数：",it, pre_mu)
        print("  混合系数：",pre_mix)

        if (sum(abs(pre_mu - last_Mu)) + sum(abs(pre_mix - last_mix))) < diff:
            break


if __name__ == '__main__':

    pre_sigma = 3.0   # 预测方差
    iternumber = 100
    diff = 0.000001
    # data = ini_data(sigma, init_mu, mix, K, N)
    # data = load("em_data.pickle")
    data = load(sys.argv[1])
    exe_em(pre_sigma,iternumber,diff, data)

    plt.hist(data[:], 100)
    plt.grid(True)
    plt.show()