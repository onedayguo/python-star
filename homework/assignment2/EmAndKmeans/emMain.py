import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 生成随机数据，4个高斯模型
def init_data_model(N, K, mix, sigma, mu0, mu1, mu2, mu3):
    global data                             # 可观测数据集
    data = np.zeros((N, 2))                 # 初始化X，N行2列。2维数据，N个样本
    data = np.mat(data)
    global predict_mu                       # 随机初始化mu1，mu2，mu3，mu4，均值
    predict_mu = np.random.random([K, 2])   # 4行2列列表，用于存储各个模型的不同维度的特征均值
    predict_mu = np.mat(predict_mu)         # 列表转化为矩阵
    global excep                            # 期望第i个样本属于第j个模型的概率的期望，i 500条，j 4个模型
    excep = np.zeros((N, 4))                # 500*4的矩阵，初始化全为0
    global predict_mix                           # 初始化混合项系数，每个模型数据所占比例
    predict_mix = [0.25, 0.25, 0.25, 0.25]       # 初始化
    a0 = mix[0]                             # 0.1
    a1 = a0 + mix[1]                        # 0.3
    a2 = a1 + mix[2]                        # 0.6
    global predict_sigma                    # 初始化协方差
    pre_sigma0 = np.mat([[5, 0], [0, 5]])
    pre_sigma1 = np.mat([[5, 0], [0, 5]])
    pre_sigma2 = np.mat([[5, 0], [0, 5]])
    pre_sigma3 = np.mat([[5, 0], [0, 5]])
    predict_sigma = np.array([pre_sigma0, pre_sigma1, pre_sigma2, pre_sigma3])
    for i in range(N):
        # 遍历N条数据
        tempnumber = np.random.random_sample()    # 生成0-1之间随机数，sigma为协方差矩阵，为方阵[[sigma_x, 0], [0, sigma_y]]
        if tempnumber < a0:                 # 0-0.1
            data[i, :] = np.random.multivariate_normal(mu0, sigma[0], 1)  # 用第一个高斯模型生成2维数据，2列数据
        elif a0 <= tempnumber < a1:                                     # 0.1-0.3
            data[i, :] = np.random.multivariate_normal(mu1, sigma[1], 1)  # 用第二个高斯模型生成2维数据，2列数据
        elif a1  <= tempnumber < a2:                                    # 0.2
            data[i, :] = np.random.multivariate_normal(mu2, sigma[2], 1)  # 用第三个高斯模型生成2维数据，2列数据
        else:
            data[i, :] = np.random.multivariate_normal(mu3, sigma[3], 1)  # 用第四个高斯模型生成2维数据，2列数据
    print("可观测数据：\n", data)                                    # 输出可观测样本,500*2
                                                                     # print("初始化的mu1，mu2，mu3，mu4：", mu)  # 输出初始化的mu
                                                                     # 由于X，mu,excep,alpha_ 是全局变量，算是initData方法返回值


def e_step( K, N):
    global data
    global predict_mu
    global excep
    global predict_mix
    global predict_sigma

    for i in range(N):              # N=5000条数据
        denom = 0                   # 分母
        for j in range(K):          # k=4,sigma.I表示矩阵的逆，np.transpose（1,0）表示矩阵转置，（0,1）表示不变
            datareshape = data[i, :].reshape(2, 1)
            mushape = predict_mu[j, :].reshape(2, 1)
            data_mu = datareshape - mushape
            denom += predict_mix[j] * math.exp((-1/2.0) * np.dot(np.dot(np.transpose(data_mu), np.linalg.inv(predict_sigma[j])),                                                data_mu))
        for j in range(K):
            datareshape = data[i, :].reshape(2, 1)
            mureshape = predict_mu[j, :].reshape(2, 1)
            data_mu = datareshape - mureshape
            numer = math.exp((-1/2.0) * np.dot(np.dot(np.transpose(data_mu), np.linalg.inv(predict_sigma[j])), data_mu))
            excep[i, j] = predict_mix[j] * numer / denom  # 求期望,第i条数据属于j模型的概率


def m_step(k, N):
    global data
    global predict_mu
    global excep
    global predict_mix
    global predict_sigma
    for j in range(0, k):
        denom = 0  # 分母
        numer = 0  # 分子
        numer_sigma = 0 # sigma 分子
        for i in range(N):
            numer += excep[i, j] * data[i, :]  # N*4  N*2  = N*2
            denom += excep[i, j]                # N*4 一个值
            numer_sigma += excep[i, j] * data[i,:] * np.transpose(data[i, :] - predict_mu[j, :]) * (data[i, :]
                                                                                                - predict_mu[j, :])
        predict_mu[j, :] = numer / denom  # 求均值 mu 4*2        N*2 / 值
        predict_mix[j] = denom / N  # 求混合项系数 1*4
        predict_sigma[j] = numer_sigma / denom                         # 4*2*2

def run_em(K,N):
    global data
    global predict_mu
    global excep
    global predict_mix
    global predict_sigma
    # 迭代执行 E步 M步
    for i in range(iter_num):
        err = 0  # 均值误差
        err_mix = 0  # 混合项系数误差
        old_mu = copy.deepcopy(predict_mu)
        old_mix = copy.deepcopy(predict_mix)
        e_step(K, N)  # E步
        m_step(K, N)         # M步
        print("迭代次数:", i + 1)
        # print("估计的均值:", mu)
        print("mu1:", predict_mu[0])
        print("mu2:", predict_mu[1])
        print("mu3:", predict_mu[2])
        print("mu4:", predict_mu[3])
        print("估计的混合项系数:", predict_mix)
        for z in range(K):
            err += (abs(old_mu[z, 0] - predict_mu[z, 0]) + abs(old_mu[z, 1] - predict_mu[z, 1]))  # 计算误差
            err_mix += abs(old_mix[z] - predict_mix[z])
        if (err <= 0.001) and (err_mix < 0.001):                                                  # 达到精度退出迭代
            print(err, err_mix)
            break


if __name__ == '__main__':
    iter_num = 100                      # 迭代次数
    N = 10000                            # 样本数目
    K = 4                               # 高斯模型数
    mu0 = [1, 5]                        # 模型0均值
    sigma0 = np.mat([[2, 0], [0, 3]])   # 模型0协方差
    mu1 = [10, 15]                      # 模型1均值
    sigma1 = np.mat([[5, 0], [0, 4]])   # 模型1协方差
    mu2 = [20, 25]                      # 模型2均值
    sigma2 = np.mat([[3, 0], [0, 5]])   # 模型2协方差
    mu3 = [35, 45]                      # 模型3均值
    sigma3 = np.mat([[5, 0], [0, 5]])   # 模型3协方差
    sigma = np.array([sigma0, sigma1, sigma2, sigma3])
    # print(sigma[0,1])
    mix = [0.1, 0.2, 0.3, 0.4]        # 模型混合项系数，模型0,1,2,3所占比例
    # init_data_model(N, K, mix, mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3)  # 生成数据 N*2 维
    init_data_model(N, K, mix, sigma, mu0, mu1, mu2, mu3)
    run_em( K, N)




    # 可视化结果
    # 画生成的原始数据
    probility = np.zeros(N)  # 混合高斯分布
    plt.subplot(221)
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), c='b', s=25, alpha=0.4, marker='o')  # T散点颜色，s散点大小，alpha透明度，marker散点形状
    plt.title('random generated data')
    # 画分类好的数据
    plt.subplot(222)
    plt.title('classified data through EM')
    order = np.zeros(N)
    color = ['b', 'r', 'k', 'y']
    for i in range(N):
        for j in range(K):
            if excep[i, j] == max(excep[i, :]):
                order[i] = j  # 选出X[i,:]属于第几个高斯模型
            probility[i] += predict_mix[j] * math.exp(
                -(data[i, :] - predict_mu[j, :]) * sigma[i].I * np.transpose(data[i, :] - predict_mu[j, :])) / (
                                        np.sqrt(np.linalg.det(sigma[i])) * 2 * np.pi)  # 计算混合高斯分布
        plt.scatter(data[i, 0], data[i, 1], c=color[int(order[i])], s=25, alpha=0.4, marker='o')  # 绘制分类后的散点图
    # 绘制三维图像
    ax = plt.subplot(223, projection='3d')
    plt.title('3d view')
    for i in range(N):
        ax.scatter(X[i, 0], X[i, 1], probility[i], c=color[int(order[i])])
    plt.show()