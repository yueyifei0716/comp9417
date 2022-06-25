from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('CarSeats.csv')

label = ['Sales']
target = data[label]

categorical_features = ['ShelveLoc', 'Urban', 'US']

numerical_data = data.drop(label + categorical_features, axis=1)

scaler = StandardScaler()
numerical_data[:] = scaler.fit_transform(numerical_data[:])

mean = numerical_data[:].mean()
var = numerical_data[:].var()

print('The means of features:\n')
print(mean, '\n')
print('The variances of features:\n')
print(var, '\n')

target = target - target.mean()
X_train, X_test, Y_train, Y_test = train_test_split(numerical_data, target, test_size=0.5, shuffle=False)

print('The first and last rows of X_train:\n')
print(X_train.head(1), '\n')
print(X_train.tail(1), '\n')

print('The first and last rows of X_test:\n')
print(X_test.head(1), '\n')
print(X_test.tail(1), '\n')

print('The first and last rows of Y_train:\n')
print(Y_train.head(1), '\n')
print(Y_train.tail(1), '\n')

print('The first and last rows of Y_test:\n')
print(Y_test.head(1), '\n')
print(Y_test.tail(1), '\n')

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

phi = 0.5
size = len(data.columns) - len(label + categorical_features)
X_t = X_train.T

B_ridge = np.linalg.inv(X_t @ X_train + phi * len(X_train) * np.identity(size)) @ X_t @ Y_train

print('The value of the ridge solution based on X_train and Y_train:\n')
print(B_ridge)

b0 = np.ones(X_train.shape[1]).reshape(7, 1)

# print(type(X_train))
# print(b0)

x = X_train.reshape(200,7)
x_t = x.T.reshape(7,200)
y = Y_train.reshape(200,1)

# print(x)
# print(x_t.shape)
# print(y)

phi = 0.5
epochs = 1000
alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

# The ridge regression loss
loss_hat = 1/len(X_train) * (np.linalg.norm(y - x @ B_ridge, ord=2) ** 2) + phi * (np.linalg.norm(B_ridge, ord=2) ** 2)

def get_beta(beta_0, x, y, x_t, alpha):
    b = b0
    betas = []
    betas.append(b)
    for k in range(1, 1001):
        b_updated = b - alpha * (-2 * (1/len(X_train)) * x_t @ (y - x @ b) + 2 * phi * b)
        b = b_updated
        betas.append(b_updated)

    return betas

def find_delta(b0, x, y, x_t,loss_hat,alpha):
    deltas = []
    betas = get_beta(b0, x, y, x_t,alpha)

    for b in betas:
        loss = 1/len(X_train) * (np.linalg.norm(y - x @ b, ord=2) ** 2) + phi * (np.linalg.norm(b, ord=2) ** 2)
        # print(loss)
        delta = loss - loss_hat
        deltas.append(delta)
    return deltas

def ridge_graph(b0, x, y, x_t, loss_hat, alpha):
    deltas = find_delta(b0, x, y, x_t, loss_hat, alpha)
    k = np.arange(0,len(deltas))

    plt.plot(k, deltas, color='blue')
    plt.xlabel("k")
    plt.ylabel("Delta")
    plt.title("alpha="+str(alpha),pad=15)

i = 1
for alpha in alphas:
    plt.subplot(3,3,i)
    ridge_graph(b0, x, y, x_t, loss_hat, alpha)
    i = i + 1

plt.tight_layout()
plt.show()

b = get_beta(b0, x, y, x_t,0.01)[1000]
train_MSE = 1/len(X_train) * (np.linalg.norm(Y_train - X_train @ b) ** 2)
print('The train MSE is :', train_MSE)

test_MSE = 1/len(X_train) * (np.linalg.norm(Y_test - X_test @ b) ** 2)
print('The test MSE is :', test_MSE)

#########################################################################

sgd_alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02]

def sgd_betas(b0, X_train,Y_train, alpha):
    b = b0
    sgd_B = []
    sgd_B.append(b)
    for i in range(1,1001):
        i = i % 200
        if i == 0:
            i = 200
        X = X_train[i-1]
        Y = Y_train[i-1]

        x = X.reshape(7,1)
        x_t = x.T

        B = b - alpha*(-2*x@(Y-x_t@b)+b)
        sgd_B.append(B)
        b = B
    return sgd_B

def sgd_deltas(b0, X_train, Y_train, loss_hat, alpha):
    y = Y_train.reshape(200,1)
    x = X_train
    sgd_del = []
    sgd_B = sgd_betas(b0, X_train,Y_train, alpha)
    for B in sgd_B:
        loss = np.linalg.norm(y - x@B, ord = 2 )**2/200 + 0.5*np.linalg.norm(B, ord = 2 )**2
        delta = loss - loss_hat
        sgd_del.append(delta)
    return sgd_del

def sgd_graph(b0, X_train, Y_train, loss_hat, alpha):
    sgd_Delta = sgd_deltas(b0, X_train, Y_train, loss_hat, alpha)
    k = np.arange(0,len(sgd_Delta))


    plt.plot(k, sgd_Delta, color='blue')
    plt.xlabel("k")
    plt.ylabel("sgd_Delta")
    plt.title("alpha="+str(alpha),pad=15)

i = 1
for alpha in sgd_alphas:
    plt.subplot(3,3,i)
    sgd_graph(b0, X_train, Y_train, loss_hat, alpha)
    i = i + 1

plt.tight_layout()
plt.show()


sgd_B =sgd_betas(b0, X_train,Y_train, 0.006)

b = sgd_B[1000]
# train_MSE = (np.linalg.norm(y - x@B_ , ord = 2)**2)/200
# test_MSE = (np.linalg.norm(y_ts - x_ts@B_ , ord = 2)**2)/200

train_MSE = 1/len(X_train) * (np.linalg.norm(Y_train - X_train @ b) ** 2)
print('The train MSE is :', train_MSE)

test_MSE = 1/len(X_train) * (np.linalg.norm(Y_test - X_test @ b) ** 2)
print('The test MSE is :', test_MSE)


### j: 全新的更新方式，每一轮只更新一个weight，假设有p个weight w(w1 w2 ... wp)， 1:03:05


### k: 实现j问算法 grad的方程

p = size        # size = the number of remaining features = 7
b = b0
for i in range(10):
    for j in range(p):
        # grad = 2/len(X_train) * ()
        b_updated = (-2 * (1/len(X_train)) * x_t @ (y - x @ b) + 2 * phi * b)



