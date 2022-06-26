from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


### Code for Question d
data = pd.read_csv('CarSeats.csv')

label = ['Sales']
target = data[label]

categorical_features = ['ShelveLoc', 'Urban', 'US']

numerical_data = data.drop(label + categorical_features, axis=1)

scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(numerical_data)

print('The means of features:')
print(numerical_data_scaled.mean(axis=0))
print()
print('The variances of features:')
print(numerical_data_scaled.var(axis=0))
print()

target_centered = (target - target.mean(axis=0)).to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(numerical_data_scaled, target_centered, test_size=0.5, shuffle=False)

print('The first and last rows of X_train:')
print(X_train[0])
print(X_train[-1])
print()
print('The first and last rows of X_test:')
print(X_test[0])
print(X_test[-1])
print()
print('The first and last rows of Y_train:')
print(Y_train[0])
print(Y_train[-1])
print()
print('The first and last rows of Y_test:')
print(Y_test[0])
print(Y_test[-1])
print()


### Code for Question e
phi = 0.5
size = len(data.columns) - len(label + categorical_features)
X_t = X_train.T

beta_hat = np.linalg.inv(X_t @ X_train + phi * len(X_train) * np.identity(size)) @ X_t @ Y_train

print('The value of the ridge solution based on X_train and Y_train:\n')
print(beta_hat)


### Code for Question g
beta_0 = np.ones(X_train.shape[1]).reshape(7, 1)

phi = 0.5
epochs = 1000
alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

def loss(X_train, Y_train, beta, phi):
    loss_beta = 1/len(X_train) * (np.linalg.norm(Y_train - X_train @ beta, ord=2) ** 2) + phi * (np.linalg.norm(beta, ord=2) ** 2)
    return loss_beta

def grad(X_train, Y_train, beta, phi):
    grad_beta = -2 * (1/len(X_train)) * X_train.T @ (Y_train - X_train @ beta) + 2 * phi * beta
    return grad_beta

def get_gd_betas(beta_0, X_train, Y_train, alpha, phi, epochs):
    beta = beta_0
    betas = []
    betas.append(beta)
    for k in range(epochs):
        b_updated = beta - alpha * grad(X_train, Y_train, beta, phi)
        beta = b_updated
        betas.append(b_updated)
    return betas

def get_gd_deltas(beta_0, X_train, Y_train, alpha, phi, epochs):
    deltas = []
    betas = get_gd_betas(beta_0, X_train, Y_train, alpha, phi, epochs)
    loss_hat = loss(X_train, Y_train, beta_hat, phi)
    for beta in betas:
        delta = loss(X_train, Y_train, beta, phi) - loss_hat
        deltas.append(delta)
    return deltas

i = 1
for alpha in alphas:
    deltas = get_gd_deltas(beta_0, X_train, Y_train, alpha, phi, epochs)
    axes = plt.subplot(3, 3, i)
    plt.plot(range(epochs + 1), deltas, color='blue')
    plt.xlabel('k')
    plt.ylabel('delta')
    plt.title('alpha = ' + str(np.format_float_positional(alpha)))
    i = i + 1
plt.suptitle('Batch GD')
plt.tight_layout()
plt.savefig('GD_plot.png', dpi=400)
plt.show()

beta_best = get_gd_betas(beta_0, X_train, Y_train, 0.005, phi, epochs)[-1]

train_MSE = 1/len(X_train) * (np.linalg.norm(Y_train - X_train @ beta_best) ** 2)
print('The train MSE is :', train_MSE)

test_MSE = 1/len(X_train) * (np.linalg.norm(Y_test - X_test @ beta_best) ** 2)
print('The test MSE is :', test_MSE)


### Code for Question h
beta_0 = np.ones(X_train.shape[1]).reshape(7, 1)
phi = 0.5
epochs = 5
sgd_alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02]

def get_sgd_betas(beta_0, X_train, Y_train, alpha, phi, epochs):
    beta = beta_0
    betas = []
    betas.append(beta)
    for i in range(epochs):
        for j in range(len(X_train)):
            x = X_train[j]
            y = Y_train[j]
            x = x.reshape(7,1)
            x_t = x.T
            b_updated = beta - alpha * (-2 * x @ ( y - x_t @ beta) + 2 * phi * beta)
            beta = b_updated
            betas.append(b_updated)
    return betas

def get_sgd_deltas(beta_0, X_train, Y_train, alpha, phi, epochs):
    deltas = []
    betas = get_sgd_betas(beta_0, X_train,Y_train, alpha, phi, epochs)
    loss_hat = loss(X_train, Y_train, beta_hat, phi)
    for beta in betas:
        delta = loss(X_train, Y_train, beta, phi) - loss_hat
        deltas.append(delta)
    return deltas

i = 1
for alpha in sgd_alphas:
    deltas = get_sgd_deltas(beta_0, X_train, Y_train, alpha, phi, epochs)
    axes = plt.subplot(3, 3, i)
    plt.plot(range(epochs * len(X_train) + 1), deltas, color='blue')
    plt.xlabel('k')
    plt.ylabel('delta')
    plt.title('alpha = ' + str(np.format_float_positional(alpha)))
    i = i + 1
plt.suptitle('SGD')
plt.tight_layout()
plt.savefig('SGD_plot.png', dpi=400)
plt.show()

beta_best = get_sgd_betas(beta_0, X_train, Y_train, 0.006, phi, epochs)[-1]

train_MSE = 1/len(X_train) * (np.linalg.norm(Y_train - X_train @ beta_best) ** 2)
print('The train MSE is :', train_MSE)

test_MSE = 1/len(X_train) * (np.linalg.norm(Y_test - X_test @ beta_best) ** 2)
print('The test MSE is :', test_MSE)


### Code for Question k
cycles = 10
p = len(X_train[0])
beta_0 = np.ones(X_train.shape[1]).reshape(7, 1)
phi = 0.5


def get_new_betas(beta_0, X_train, Y_train, phi):
    beta = beta_0
    betas = []
    betas.append(beta)
    y = Y_train
    for i in range(cycles):
        for j in range(p):
            x_j = X_train[:, j].reshape(1, len(X_train))
            x_nj = np.delete(X_train, j, 1)
            beta_nj = np.delete(beta, j, 0)
            bj_updated = (x_j @ y - x_j @ x_nj @ beta_nj) / (x_j @ x_j.T + len(X_train) * phi)
            beta = copy.deepcopy(beta)
            beta[j] = bj_updated
            betas.append(beta)
    return betas


def get_new_deltas(beta_0, X_train, Y_train, phi):
    deltas = []
    betas = get_new_betas(beta_0, X_train,Y_train, phi)
    loss_hat = loss(X_train, Y_train, beta_hat, phi)
    for beta in betas:
        delta = loss(X_train, Y_train, beta, phi) - loss_hat
        deltas.append(delta)
    return deltas


deltas = get_new_deltas(beta_0, X_train, Y_train, phi)

plt.plot(range(cycles * p + 1), deltas, color='blue', label='new algorithm')
plt.plot(range(cycles * p + 1), get_gd_deltas(beta_0, X_train, Y_train, 0.005, phi, 1000)[0:cycles * p + 1], color='green', label='batch GD')
plt.plot(range(cycles * p + 1), get_sgd_deltas(beta_0, X_train, Y_train, 0.006, phi, 5)[0:cycles * p + 1], color='orange', label='SGD')
plt.xlabel('k')
plt.legend(loc='upper right')                           # creates legend in top right corner of plot
plt.ylabel('delta')
plt.title('New algorithem compared with GD and SGD')
plt.savefig('comparison.png', dpi=400)
plt.show()

beta_best = get_new_betas(beta_0, X_train, Y_train, phi)[-1]

train_MSE = 1/len(X_train) * (np.linalg.norm(Y_train - X_train @ beta_best) ** 2)
print('The train MSE is :', train_MSE)

test_MSE = 1/len(X_train) * (np.linalg.norm(Y_test - X_test @ beta_best) ** 2)
print('The test MSE is :', test_MSE)
