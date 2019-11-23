import scipy.io as scio
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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


if __name__ == "__main__":
    dataA_X, dataA_Y, dataB_X, dataB_Y, dataC_X, dataC_Y = get_data_from_matfile()

    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0,)
    gmm.fit(dataC_X)
    label = gmm.predict(dataC_X)
    print(label)
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
    plt.title("dataC_X-EM-k=4 ")
    plt.grid(True)
    plt.show()