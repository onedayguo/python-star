import numpy as np
import matplotlib.pyplot as plt
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

x_mean = np.mean(x)
y_mean = np.mean(y)
num = 0.0  # 分子
d = 0.0  # 分母
for x_i, y_i in  zip(x, y):
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2
a = num / d
b = y_mean - a * x_mean
print(a)
print(b)
y_hat = a * x + b
# plt.scatter(x, y)
# plt.plot(x, y_hat, color='red')
# plt.axis([0, 6, 0, 6])
# plt.show()
from note.linear.SimpleLinearRegression import SimpleLinearRegression1
reg1 = SimpleLinearRegression1()
reg1.fit(x, y)


