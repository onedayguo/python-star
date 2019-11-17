# coding:gbk
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


def ini_data(sigma, init_mu, mix, N):
    data = np.zeros((1, N))                # 1��N�еĵľ������硾0,0,0,0,0,0��...��
    for i in range(N):
        temp = np.random.random(1)
        if temp < mix[0]:
            data[0, i] = np.random.normal() * sigma[0] + init_mu[0]
        else:
            data[0, i] = np.random.normal() * sigma[1] + init_mu[1]
    return data


def exe_em(sigma, init_mu, mix, N, iter, diff, pre_expec0, pre_mix0, pre_sigma0, pre_mu0):

    data = ini_data(sigma, init_mu, mix, N)

    K = len(mix)              # ���ϵ�����������������
    pre_expec = pre_expec0    # ���е�����������
    pre_mix = pre_mix0        # ���е����Ļ��ϵ��
    pre_sigma = pre_sigma0    # ���е����ķ���
    pre_mu = pre_mu0          # ���е����ľ�ֵ

    # EM����

    for it in range(iter):
        old_Mu = copy.deepcopy(pre_mu)

        # e step��������������
        for i in range(N):
            down = 0
            for j in range(K):
                # ���ڼ�����������ķ�ĸ����
                d_u = float(data[0, i]-pre_mu[j])
                down += math.exp(-1 / 2.0 * math.pow(d_u / 3.0, 2))
            for j in range(K):
                # ���ڼ�����������ķ��Ӳ���
                d_u = float(data[0, i] - pre_mu[j])
                up = math.exp(-1 / 2.0 * math.pow(d_u / 3.0, 2))
                pre_expec[i, j] = up / down   # ���µ�i���������ڵ�j��ģ�͵ĸ��ʣ������ݵĸ����ܶȺ���

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

        print(it, "Ԥ���ֵ ", pre_mu)
        print("Ԥ�ⷽ�� ", pre_sigma)
        print("Ԥ����ϵ�� ", pre_mix)
        # if  < diff:
        #    break


if __name__ == '__main__':
    sigma = [3.0, 3.0]  # ��ʼ�����ݷ��һά����
    init_mu = [25, 65]  # ��ʼ�����ݾ�ֵ
    mix = [0.6, 0.4]  # ���ϵ��
    K = 2           # �������࣬��Ҫ�ͷ����ֵ������Ӧ
    N = 10000       # ���������ܺ�
    iter = 100      # ����������
    diff = 0.001    # ��ֵ���������ֲ�����ֵ

    pre_expec0 = np.zeros([N, K])  # �����������飬���壺��N���������ڵ�K��ģ�͵ĸ���
    pre_expec0 = np.mat(pre_expec0)
    pre_mix0 = [0.5, 0.5]          # Ԥ��ģ�ͻ��ϵ����ʼ�������壺ÿ��ģ���µ�������ռ����
    pre_sigma0 = [5.0, 5.0]        # Ԥ�ⷽ���ʼ�� �����壺ÿ��ģ�͵ķ���
    pre_mu0 = np.array([10.0, 10.0])        # Ԥ���ֵ��ʼ�� �����壺ÿ��ģ�͵ľ�ֵ

    exe_em(sigma, init_mu, mix, N, iter, diff, pre_expec0,pre_mix0, pre_sigma0, pre_mu0)

    # print(alpha)
    # plt.hist(data[0, :], 100)
    # plt.grid(True)
    # plt.show()