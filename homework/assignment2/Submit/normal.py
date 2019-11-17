from numpy import *
import math
import copy
import numpy as np
import pickle
import sys


def ini_data(sigma, init_mu, mix, N):
    data = []                # 1行N列的的矩阵，型如【0,0,0,0,0,0，...】
    for i in range(0, N):
        temp = np.random.random(1)
        if temp < mix[0]:
            data.append(np.random.normal() * sigma[0] + init_mu[0])
        else:
            data.append(np.random.normal() * sigma[1] + init_mu[1])
    return data


if __name__ == "__main__":
    sigma = [5.0, 3.0]  # 初始化数据方差，一维数据
    init_mu = [5, 65]  # 初始化数据均值
    mix = [0.7, 0.3]  # 混合系数
    K = 2  # 数据种类，需要和方差均值数量对应
    N = 50000  # 数据数量总和

    data_kmeans = ini_data(sigma,init_mu,mix,N)
    data_em = ini_data(sigma,init_mu,mix,N)

    with open('kmeans_data.pickle', 'wb') as kmeans:
        pickle.dump(data_kmeans, kmeans)
    with open('em_data.pickle', 'wb') as em:
        pickle.dump(data_em, em)
    print("数据生成成功,文件名为：kmeans_data.pickle,em_data.pickle")