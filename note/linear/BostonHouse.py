import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 波士顿房产数据
boston = datasets.load_boston()
x = boston.data[:, 5]
print(x.shape)
y = boston.target
print(y.shape)
x = x[y < 50.0]
y = y[y < 50.0]

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)
plt.scatter(x, y)
plt.show()
