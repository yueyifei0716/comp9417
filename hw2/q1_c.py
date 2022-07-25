import numpy as np


def grad(x, y):
    dx = -400 * x * (y - x ** 2) - 2 * (1 - x)
    dy = 200 * (y - x ** 2)
    return np.transpose(np.array([dx, dy]))

def hess(x, y):
    h00 = -400 * (y - 3 * x ** 2) + 2
    h01 = -400 * x
    h10 = -400 * x
    h11 = 200
    return np.array([[h00, h01], [h10, h11]])

x_list = []
x0 = np.transpose(np.array([-1.2, 1]))
x_list.append(x0)

x = x0
k = 1

while True:
    if np.linalg.norm(grad(x[0], x[1]), ord=2) <= 10 ** -6:
        break
    new_x = x - np.matmul(np.linalg.inv(hess(x[0], x[1])), grad(x[0], x[1]))
    x = new_x
    x_list.append(new_x)
    k = k + 1

for i in range(0, len(x_list)):
    x = x_list[i]
    print("k = {}, x({}) = [{},{}]".format(i, i, x[0], x[1]))
