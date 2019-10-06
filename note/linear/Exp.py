import numpy as np
import matplotlib as plt

lambd = 0.5
x = np.arange(0,15,0.1)
y = lambd*np.exp(-lambd*x)
plt.plot(x,y)
plt.title('Exponential:$\lambda$=%.2f' % lambd)
plt.xlabel('x')
plt.ylabel('Probability density')
plt.show()
