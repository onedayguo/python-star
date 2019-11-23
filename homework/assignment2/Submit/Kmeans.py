
import scipy.io as scio
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import metrics  #导入sklearn效果评估模块


def get_data_from_matfile():
    data_path = "E:/CS5487/Assignment2/PA2-cluster-data/cluster_data.mat"
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
    x_distance = np.power(point1[0] - point2[0], 2)
    y_distance = np.power(point1[1] - point2[1], 2)
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
    index = n_random_index(k_cluster, 0, data_x[0])  # generate k the index of points
    # get the k  random center points from data_x
    center = []
    cluster = [[]]
    for k in range(k_cluster):
        center.append(dataA_X[index.pop()])
    for i in range(iteration):
        for point in data_x:
            min_dis = 0
            for center_k in center:
                dis = distance(point, center_k)
                min_dis = min(min_dis, dis)




if __name__ == "__main__":
    dataA_X, dataA_Y, dataB_X, dataB_Y, dataC_X, dataC_Y = get_data_from_matfile()
    estimator = KMeans(n_clusters=4)
    # 预测类别标签结果 # fit_predict表示拟合+预测，也可以分开写
    res = estimator.fit_predict(dataC_X)
    label = estimator.labels_
    # 各个类别的聚类中心值
    centroids = estimator.cluster_centers_
    # # 聚类中心均值向量的总和
    inertia = estimator.inertia_
    row = len(dataC_X)
    for i in range(row):
        if int(label[i]) == 0:
            plt.scatter(dataC_X[i][0], dataC_X[i][1], marker='*', color='red')
        if int(label[i]) == 1:
            plt.scatter(dataC_X[i][0], dataC_X[i][1], marker='o', color='black')
        if int(label[i]) == 2:
            plt.scatter(dataC_X[i][0], dataC_X[i][1], marker='+', color='blue')
        if int(label[i]) == 3:
            plt.scatter(dataC_X[i][0], dataC_X[i][1], marker='^', color='green')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("dataC_X-Kmeans-k=4 ")
    plt.grid(True)
    plt.show()
