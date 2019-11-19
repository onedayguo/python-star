#made by kami
#2019/4/20
# from numpy import *
import math
import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys


def ini_kmeans_data(sigma, init_mu, mix, N):
    data = []                # 1行N列的的矩阵，型如【0,0,0,0,0,0，...】
    for i in range(0, N):
        temp = np.random.random(1)
        if temp < mix[0]:
            data.append(np.random.normal() * sigma[0] + init_mu[0])
        else:
            data.append(np.random.normal() * sigma[1] + init_mu[1])
    return data


# 求点之间距离
def distance(point1, point2):
    return np.sqrt(np.power(point1 - point2, 2))


def new_center(data_array):
    lenth = len(data_array)
    sum = 0
    for i in range(lenth):
        sum += data_array[i]
    center = sum / lenth
    return center


# K均值聚类算法实现，dataSet数据集，K聚簇数，
def KMeans(data,iter_number,diff):

    # 选初始质心,随机选择点的下标
    index_center = []
    N = len(data)
    index_center.append(np.random.randint(0, N-1))
    index_center.append(np.random.randint(0, N-1))
    dataA = []
    dataB = []

    for i in range(N):
        disA = distance(data[i], data[index_center[0]])  # 距离质心A的距离
        disB = distance(data[i], data[index_center[1]])  # 距离质心B的距离
        if disA <= disB:
            dataA.append(data[i])
        else:
            dataB.append(data[i])

    last_A_center = 0.0
    last_B_center = 0.0
    it_1 = 0  # 用于统计迭代次数
    for it in range(iter_number):
        new_A_center = new_center(dataA)
        new_B_center = new_center(dataB)

        dataA.clear()
        dataB.clear()
        for i in range(N):
            disA = distance(data[i], new_A_center)
            disB = distance(data[i], new_B_center)
            if disA <= disB:
                dataA.append(data[i])
            else:
                dataB.append(data[i])

        if (abs(new_A_center - last_A_center)+abs(new_B_center-last_B_center)) < diff:
            print("迭代次数：", it+1, "均值A：", np.mean(dataA), "方差：",np.std(dataA), "混合系数：", len(dataA) / N)
            print("        ：", it+1, "均值B：", np.mean(dataB), "方差：",np.std(dataB), "混合系数：", len(dataB) / N)
            return dataA,dataB,it+1

        last_A_center = copy.deepcopy(new_A_center)
        last_B_center = copy.deepcopy(new_B_center)
        it_1 =it+1
        print("迭代次数：",it+1,"均值A：",np.mean(dataA),"方差：","混合系数：",len(dataA) / N)
        print("        ：",it+1, "均值B：", np.mean(dataB), "方差：", "混合系数：", len(dataB) / N)
    return dataA,dataB,it_1


if __name__ == "__main__":
    sigma = [5.0, 3.0]  # 初始化数据方差，一维数据
    init_mu = [5, 65]  # 初始化数据均值
    mix = [0.7, 0.3]  # 混合系数
    K = 2  # 数据种类，需要和方差均值数量对应


    iter_number = 100
    diff = 0.00001

    data = np.load(sys.argv[1])
    N = len(data)
    dataA , dataB, iter_final = KMeans(data,iter_number,diff)

    mixA = len(dataA) / N
    mixB = len(dataB) / N
    meanA = np.mean(dataA)
    meanB = np.mean(dataB)
    stdA = np.std(dataA)
    stdB = np.std(dataB)
    print("最终迭代次数与结果：",iter_final)
    print("均值A：", meanA, "方差：",stdA, "混合系数：", mixA)
    print("均值B：", meanB, "方差：", stdB,"混合系数：", mixB)
    if init_mu[0]<=init_mu[1]:
        if meanA<=meanB:
            print("accuracy: ",(init_mu[0]-abs(meanA-init_mu[0])) / init_mu[0])
        else:
            print("accuracy: ", (init_mu[0]-abs(meanB - init_mu[0])) / init_mu[0])
    if init_mu[1] < init_mu[0]:
        if meanA <= meanB:
            print("accuracy: ", (init_mu[1]-abs(meanA - init_mu[1])) / init_mu[1])
        else:
            print("accuracy: ", (init_mu[1]-abs(meanB - init_mu[1])) / init_mu[1])

    plt.hist(data[:], 100)
    plt.grid(True)
    plt.show()

