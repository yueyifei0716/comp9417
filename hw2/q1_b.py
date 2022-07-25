import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


def func(x, y):
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


# create two one-dimensional grids using linspace
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
# combine the two one-dimensional grids into one two-dimensional grid
X, Y = np.meshgrid(x, y)

Z = func(X, Y)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.savefig('qb.png', dpi=400)
plt.show()
