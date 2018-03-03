import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')

w1 = np.array([0, 1])
w2 = np.array([1, 1])
w3 = np.array([5, 1])
w4 = np.array([0, -2])
w5 = np.array([-2, 5])

x1 = np.arange(0, 1, 0.01)
x2 = np.arange(0, 1, 0.01)
X1, X2 = np.meshgrid(x1, x2)

f1 = 1 / (2*np.sum(abs(w4))) * np.cos( w4[0] * X1 + w4[1] * X2 )
print(X1)
print(X2)
print(f1)

surf = ax.plot_surface(X1, X2, f1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()