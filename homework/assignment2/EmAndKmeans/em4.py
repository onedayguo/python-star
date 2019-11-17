import math
import copy
import numpy as np
import matplotlib.pyplot as plt

sigma = np.mat([[30, 0], [0, 30]])  # 协方差矩阵，计算多维方差
m = sigma.I
x = np.arange(4,8).reshape((2,2))
#print(x)
#print(x.transpose((1,0)))
#a = np.reshape(np.arange(6),(2,3))
a = np.reshape(np.arange(20),(5,2,2))
b=np.linalg.det(a)
print(a)
print(b)
