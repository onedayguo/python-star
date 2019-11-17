# coding:gbk
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


# ָ��k����˹�ֲ�����������ָ��k=2��ע��2����˹�ֲ�������ͬ������Sigma���ֱ�ΪMu1,Mu2��
def ini_data(Sigma, Mu1, Mu2, k, N):
    global data
    global Mu
    global Expectations
    global alpha

    alpha = [0.5, 0.5]
    data = np.zeros((1, N))                # 1��N�еĵľ������硾0,0,0,0,0,0��...��
    Mu = np.random.random(2)             # ����2��0-1֮������������0.451256521 , 0.8652145965235
    print(type(Mu))
    Expectations = np.zeros((N, k))     # N��K�еľ��󣬼�¼�쳣������k��2��N��2��
    for i in range(0, N):
        if np.random.random(1) > 0.7:
            # random.normalvariate(5, 1)
            data[0, i] = np.random.normal() * Sigma + Mu1
        else:
            data[0, i] = np.random.normal() * Sigma + Mu2


# EM�㷨������1������E[zij]
def e_step(Sigma, k, N):
    global Expectations
    global Mu
    global data
    global alpha
    for i in range(0, N):
        Denom = 0
        for j in range(0, k):
            Denom += math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(data[0, i] - Mu[j])) ** 2)   #��̬�ֲ������ܶȺ�����exp����
        for j in range(0, k):
            Numer = math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(data[0, i] - Mu[j])) ** 2)
            Expectations[i, j] = Numer / Denom


# EM�㷨������2�������E[zij]�Ĳ���Mu
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

# �㷨����iter_num�Σ���ﵽ����Epsilonֹͣ����
def run(Sigma, Mu1, Mu2, k, N, iter_num, Epsilon):
    global Expectations
    global data
    global alpha
    ini_data(Sigma, Mu1, Mu2, k, N)
    print(u"��ʼ<u1,u2>:", Mu)
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