from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#### d
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


##### e
phi = 0.5
size = len(data.columns) - len(label + categorical_features)
X_t = X_train.T

beta_hat = np.linalg.inv(X_t @ X_train + phi * len(X_train) * np.identity(size)) @ X_t @ Y_train

print('The value of the ridge solution based on X_train and Y_train:\n')
print(beta_hat)


#### g

beta_0 = np.ones(X_train.shape[1]).reshape(7, 1)
# X_train_t = X_train.T

phi = 0.5
epochs = 1000
alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

def loss(X_train, Y_train, beta, phi):
    loss_beta = 1/len(X_train) * (np.linalg.norm(Y_train - X_train @ beta, ord=2) ** 2) + phi * (np.linalg.norm(beta, ord=2) ** 2)
    return loss_beta

def grad(X_train, Y_train, beta, phi):
    grad_beta = -2 * (1/len(X_train)) * X_train.T @ (Y_train - X_train @ beta) + 2 * phi * beta
    return grad_beta

def get_betas(beta_0, X_train, Y_train, alpha, phi):
    beta = beta_0
    betas = []
    betas.append(beta)
    for k in range(epochs):
        b_updated = beta - alpha * grad(X_train, Y_train, beta, phi)
        beta = b_updated
        betas.append(b_updated)
    return betas

def get_deltas(beta_0, X_train, Y_train, alpha, phi):
    deltas = []
    betas = get_betas(beta_0, X_train, Y_train, alpha, phi)
    loss_hat = loss(X_train, Y_train, beta_hat, phi)
    for beta in betas:
        delta = loss(X_train, Y_train, beta, phi) - loss_hat
        deltas.append(delta)
    return deltas

i = 1
for alpha in alphas:
    deltas = get_deltas(beta_0, X_train, Y_train, alpha, phi)
    axes = plt.subplot(3, 3, i)
    plt.plot(range(epochs + 1), deltas, color='blue')
    plt.xlabel('k')
    plt.ylabel('delta')
    plt.title('alpha = ' + str(np.format_float_positional(alpha)))
    i = i + 1
plt.tight_layout()
plt.savefig("GB plot.png", dpi=400)
# plt.show()

beta_best = get_betas(beta_0, X_train, Y_train, 0.005, phi)[-1]

train_MSE = 1/len(X_train) * (np.linalg.norm(Y_train - X_train @ beta_best) ** 2)
print('The train MSE is :', train_MSE)

test_MSE = 1/len(X_train) * (np.linalg.norm(Y_test - X_test @ beta_best) ** 2)
print('The test MSE is :', test_MSE)


#### h

# #########################################################################

beta_0 = np.ones(X_train.shape[1]).reshape(7, 1)
phi = 0.5
epochs = 5
sgd_alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02]

def get_sgd_betas(beta_0, X_train, Y_train, alpha, phi):
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

def get_sgd_deltas(beta_0, X_train, Y_train, alpha, phi):
    deltas = []
    betas = get_sgd_betas(beta_0, X_train,Y_train, alpha, phi)
    loss_hat = loss(X_train, Y_train, beta_hat, phi)
    for beta in betas:
        delta = loss(X_train, Y_train, beta, phi) - loss_hat
        deltas.append(delta)
    return deltas

i = 1
for alpha in sgd_alphas:
    deltas = get_sgd_deltas(beta_0, X_train, Y_train, alpha, phi)
    axes = plt.subplot(3, 3, i)
    plt.plot(range(epochs * len(X_train) + 1), deltas, color='blue')
    plt.xlabel('k')
    plt.ylabel('delta')
    plt.title('alpha = ' + str(np.format_float_positional(alpha)))
    i = i + 1
plt.tight_layout()
plt.savefig("SGB plot.png", dpi=400)
# plt.show()


beta_best = get_sgd_betas(beta_0, X_train, Y_train, 0.006, phi)[-1]

train_MSE = 1/len(X_train) * (np.linalg.norm(Y_train - X_train @ beta_best) ** 2)
print('The train MSE is :', train_MSE)

test_MSE = 1/len(X_train) * (np.linalg.norm(Y_test - X_test @ beta_best) ** 2)
print('The test MSE is :', test_MSE)


# ### j: 全新的更新方式，每一轮只更新一个weight，假设有p个weight w(w1 w2 ... wp)， 1:03:05


# ### k: 实现j问算法 grad的方程

# p = size        # size = the number of remaining features = 7
# b = b0
# for i in range(10):
#     for j in range(p):
#         # grad = 2/len(X_train) * ()
#         b_updated = (-2 * (1/len(X_train)) * x_t @ (y - x @ b) + 2 * phi * b)



