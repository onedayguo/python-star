# coding:gbk
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


# 指定k个高斯分布参数，这里指定k=2。注意2个高斯分布具有相同均方差Sigma，分别为Mu1,Mu2。
def ini_data(Sigma, Mu1, Mu2, k, N):
    global data
    global Mu
    global Expectations
    global alpha

    alpha = [0.5, 0.5]
    data = np.zeros((1, N))                # 1行N列的的矩阵，型如【0,0,0,0,0,0，...】
    Mu = np.random.random(2)             # 生成2个0-1之间的随机数，如0.451256521 , 0.8652145965235
    print(type(Mu))
    Expectations = np.zeros((N, k))     # N行K列的矩阵，记录异常，这里k是2，N行2列
    for i in range(0, N):
        if np.random.random(1) > 0.7:
            # random.normalvariate(5, 1)
            data[0, i] = np.random.normal() * Sigma + Mu1
        else:
            data[0, i] = np.random.normal() * Sigma + Mu2


# EM算法：步骤1，计算E[zij]
def e_step(Sigma, k, N):
    global Expectations
    global Mu
    global data
    global alpha
    for i in range(0, N):
        Denom = 0
        for j in range(0, k):
            Denom += math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(data[0, i] - Mu[j])) ** 2)   #正态分布概率密度函数的exp部分
        for j in range(0, k):
            Numer = math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(data[0, i] - Mu[j])) ** 2)
            Expectations[i, j] = Numer / Denom


# EM算法：步骤2，求最大化E[zij]的参数Mu
def m_step(k, N):
    global Expectations
    global data
    global alpha
    for j in range(k):
        Numer = 0
        Denom = 0
        for i in range(0, N):
            Numer += Expectations[i, j] * data[0, i]
            Denom += Expectations[i, j]
        Mu[j] = Numer / Denom
        alpha[j] = Denom / N

# 算法迭代iter_num次，或达到精度Epsilon停止迭代
def run(Sigma, Mu1, Mu2, k, N, iter_num, Epsilon):
    global Expectations
    global data
    global alpha
    ini_data(Sigma, Mu1, Mu2, k, N)
    print(u"初始<u1,u2>:", Mu)
    for i in range(iter_num):
        old_Mu = copy.deepcopy(Mu)
        e_step(Sigma, k, N)
        m_step(k, N)
        print(i, Mu)
        if sum(abs(Mu - old_Mu)) < Epsilon:
            break
if __name__ == '__main__':
    run(6, 25, 65, 2, 50000, 100, 0.0001)
    print(alpha)
    plt.hist(data[0, :], 100)
    plt.grid(True)
    plt.show()