import numpy as np

alpha = 0.1
gamma = 0.2

A = np.array([[1,2,1,-1], [-1,1,0,2], [0,-1,-2,1]])     # A: 3 x 4
b = np.array([[3], [2], [-2]])                          # b: 3 x 1
A_t = A.T                                               # A_t: 4 x 3
x0 = np.array([1,1,1,1]).reshape(4, 1)                  # x_0: 1 x 4 -> 4 x 1

x_list = []
x_list.append(x0)

x = x0
k = 1

while True:
    grad = A_t @ (A @ x - b) + gamma * x
    if np.linalg.norm(grad, ord=2) < 0.001:
        break
    new_x = x - alpha * grad
    x = new_x
    x_list.append(new_x)
    k = k + 1

length = len(x_list)

for i in range(0, 6):
    x = x_list[i].reshape(1, 4)[0]
    print("k = {}, x(k) = [{},{},{},{}]".format(i,round(x[0],4) ,round(x[1],4), round(x[2],4), round(x[3],4)))

for j in range(length - 5, length):
    x = x_list[j].reshape(1, 4)[0]
    print("k = {}, x(k) = [{},{},{},{}]".format(j,round(x[0],4) ,round(x[1],4), round(x[2],4), round(x[3],4)))
