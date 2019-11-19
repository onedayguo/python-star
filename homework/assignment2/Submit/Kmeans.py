import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import scipy.io as scio
import matplotlib.pyplot as plt
import random

def get_data_from_matfile():
    data_path="E:/CS5487/Assigment2/cluster_data.mat"
    data = scio.loadmat(data_path)
    dataA_X = np.transpose(data['dataA_X'])  # (200,2)
    dataA_Y = np.transpose(data['dataA_Y'])  # (200,1)
    dataB_X = np.transpose(data['dataB_X'])  # (200,2)
    dataB_Y = np.transpose(data['dataB_Y'])  # (200,1)
    dataC_X = np.transpose(data['dataC_X'])  # (200,2)
    dataC_Y = np.transpose(data['dataB_Y'])  # (200,1)
    return dataA_X, dataA_Y, dataB_X, dataB_Y, dataC_X, dataC_Y



# 求二维点之间距离
def distance(point1, point2):
    x_distance = np.power(point1[0] - point2[0],2)
    y_distance = np.power(point1[1] - point2[1],2)
    direct_distance = np.sqrt(x_distance + y_distance)
    return direct_distance


def new_center(data_array):
    lenth = len(data_array)
    sum = 0
    for i in range(lenth):
        sum += data_array[i]
    center = sum / lenth
    return center


def n_random_index(n, low, high):
    index = set()
    while len(index) <= n:
        index.add(np.random.randint(low, high))
    return index


def k_means(data_x, data_y, k_cluster, iteration, error):
    index = n_random_index(k_cluster, data_x[0])  # generate k the index of points


if __name__ == "__main__":
    dataA_X, dataA_Y, dataB_X, dataB_Y, dataC_X, dataC_Y = get_data_from_matfile()
    print('dataA_X维度：', dataA_X.shape)
    print('dataA_Y维度：', dataA_Y.shape)
    print('dataB_X维度：', dataB_X.shape)
    print('dataB_Y维度：', dataB_Y.shape)
    print('dataC_X维度：', dataC_X.shape)
    print('dataC_Y维度：', dataC_Y.shape)

