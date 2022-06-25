import torch
import torch.nn as nn
from torch import optim

gamma = 0.2
alpha = 0.1

A = torch.tensor([[1., 2., 1., -1.], [-1., 1., 0., 2.], [0., -1., -2., 1.]])
b = torch.tensor([[3.], [2.], [-2.]])
A_t = A.t()
x0 = torch.tensor([[1.], [1.], [1.], [1.]])

x_dict = {}
x_dict[0] = x0.tolist()
x = x0.clone()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(x, requires_grad=True)
    def forward(self, x):
        z = torch.mul(torch.pow(torch.norm(torch.sub(torch.mm(A, x), b)), 2), 1/2)
        w = torch.mul(torch.pow(torch.norm(x), 2), gamma/2)
        return torch.add(z, w)

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=alpha)
terminationCond = False

k = 1

while not terminationCond:
    optimizer.zero_grad()
    loss = model(model.x)
    loss.backward()
    optimizer.step()

    if torch.norm(model.x.grad) < 0.001:
        terminationCond = True
    else:
        x_dict[k] = model.x.tolist()
        k = k + 1

length = len(x_dict)

for i in range(0, 6):
    x = x_dict[i]
    print("k = {}, x(k) = [{},{},{},{}]".format(i, round(x[0][0], 4) ,round(x[1][0], 4), round(x[2][0], 4), round(x[3][0], 4)))


for j in range(length - 5, length):
    x = x_dict[j]
    print("k = {}, x(k) = [{},{},{},{}]".format(j, round(x[0][0], 4) ,round(x[1][0], 4), round(x[2][0], 4), round(x[3][0], 4)))