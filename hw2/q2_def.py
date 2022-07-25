import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helper import sigmoid, loss
import matplotlib.pyplot as plt

# Question d of Part 2
data = pd.read_csv('songs.csv')

# (I)
remove_features = ['Artist Name', 'Track Name', 'key', 'mode', 'time_signature', 'instrumentalness']
data = data.drop(remove_features, axis=1)

# (II)
data = data[(data['Class'] == 5) | (data['Class'] == 9)]
data['Class'].replace([5, 9], [1, 0], inplace=True)
data = data.reset_index(drop=True)

# (III)
data = data.dropna()

# (IV)
X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:, 0:-1], data.iloc[:, -1], test_size=0.3,
                                                    random_state=23)
# (V)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# (VI)
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

print('The first row of X_train:')
print(X_train[0][:3])
print('The last row of X_train:')
print(X_train[-1][:3])
print()
print('The first row of X_test:')
print(X_test[0][:3])
print('The last row of X_test:')
print(X_test[-1][:3])
print()
print('The first row of Y_train:')
print(Y_train[0])
print('The last row of Y_train:')
print(Y_train[-1])
print()
print('The first row of Y_test:')
print(Y_test[0])
print('The last row of Y_test:')
print(Y_test[-1])
print()


# Question e of Part 2

epochs = 60
lam = 0.5
beta_0 = 0
beta = np.ones(len(data.columns) - 1)
gamma_0 = np.insert(beta, 0, beta_0)
a = 0.5
b = 0.8


def grad(gamma, X, y, lam):
    z = np.dot(X, gamma[1:]) + gamma[0]
    grad_beta = gamma[1:] + (lam / len(X)) * np.dot((sigmoid(z) - y), X)
    grad_beta0 = (lam / len(X)) * np.sum((sigmoid(z) - y))
    grad_gamma = np.insert(grad_beta, 0, grad_beta0)
    return grad_gamma


def get_gd_gammas_alphas(gamma_0, X, y, epochs, lam):
    alphas = []
    gammas = []
    gamma = gamma_0
    alphas.append(1)
    gammas.append(gamma)
    # Backtracking line search
    for k in range(epochs):
        alpha = 1
        if loss(gamma - alpha * grad(gamma, X, y, lam), X, y, lam) > loss(gamma, X, y, lam) - a * alpha * np.linalg.norm(grad(gamma, X, y, lam), ord=2)**2:
            alpha = alpha * b
        alphas.append(alpha)
        gamma_updated = gamma - alpha * grad(gamma, X, y, lam)
        gamma = gamma_updated
        gammas.append(gamma_updated)
    return gammas, alphas


def get_gd_losses(gamma_0, X, y, epochs, lam):
    losses = []
    gammas = get_gd_gammas_alphas(gamma_0, X, y, epochs, lam)[0]
    for gamma in gammas:
        gamma_loss = loss(gamma, X, y, lam)
        losses.append(gamma_loss)
    return losses


alphas = get_gd_gammas_alphas(gamma_0, X_train, Y_train, epochs, lam)[1]
losses_train = get_gd_losses(gamma_0, X_train, Y_train, epochs, lam)
losses_test = get_gd_losses(gamma_0, X_test, Y_test, epochs, lam)
print('Final loss achieved by GD algorithm on the train data')
print(losses_train[-1])
print('Final loss achieved by GD algorithm on the test data')
print(losses_test[-1])

plt.plot(range(epochs), alphas[1:], color='blue', label='GD step size')
plt.xlabel('epoch')
plt.legend()  # creates legend in top right corner of plot
plt.ylabel('step size')
plt.title('Plot of step size vs epoch')
plt.savefig('qe_step_sizes.png', dpi=400)
plt.show()

plt.plot(range(epochs), losses_train[1:], color='blue', label='GD train loss')
plt.xlabel('epoch')
plt.legend()  # creates legend in top right corner of plot
plt.ylabel('train loss')
plt.title('Plot of train loss vs epoch')
plt.savefig('qe_train_loss.png', dpi=400)
plt.show()


# Question f of Part 2

def hess(gamma, X, lam):
    # H = I + X.T * S * X
    hess_matrix = np.zeros((11, 11))
    identity_matrix = np.identity(11)
    z = np.dot(X, gamma[1:]) + gamma[0]
    diagnose_matrix = np.diag((lam / len(X)) * sigmoid(z) * (1 - sigmoid(z)))
    col = np.ones((len(X), 1))
    X = np.column_stack((col, X))
    hess_matrix = identity_matrix + X.T @ diagnose_matrix @ X
    return hess_matrix


def get_dn_gammas_alphas(gamma_0, X, y, epochs, lam):
    alphas = []
    gammas = []
    gamma = gamma_0
    alphas.append(1)
    gammas.append(gamma)
    # Backtracking line search
    for k in range(epochs):
        alpha = 1
        if loss(gamma - alpha * grad(gamma, X, y, lam), X, y, lam) > loss(gamma, X, y, lam) - a * alpha * np.linalg.norm(grad(gamma, X, y, lam), ord=2)**2:
            alpha = alpha * b
        alphas.append(alpha)
        gamma_updated = gamma - alpha * np.matmul(np.linalg.inv(hess(gamma, X, lam)), grad(gamma, X, y, lam))
        gamma = gamma_updated
        gammas.append(gamma_updated)
    return gammas, alphas


def get_dn_losses(gamma_0, X, y, epochs, lam):
    losses = []
    gammas = get_dn_gammas_alphas(gamma_0, X, y, epochs, lam)[0]
    for gamma in gammas:
        gamma_loss = loss(gamma, X, y, lam)
        losses.append(gamma_loss)
    return losses


alphas = get_dn_gammas_alphas(gamma_0, X_train, Y_train, epochs, lam)[1]
losses_train_dn = get_dn_losses(gamma_0, X_train, Y_train, epochs, lam)
losses_test_dn = get_dn_losses(gamma_0, X_test, Y_test, epochs, lam)
print('Final loss achieved by Newton algorithm on the train data')
print(losses_train_dn[-1])
print('Final loss achieved by Newton algorithm on the test data')
print(losses_test_dn[-1])

plt.plot(range(epochs), alphas[1:], color='red', label='Newton step size')
plt.xlabel('epoch')
plt.legend()  # creates legend in top right corner of plot
plt.ylabel('step size')
plt.title('Plot of step size vs epoch')
plt.savefig('qf_step_sizes.png', dpi=400)
plt.show()

plt.plot(range(epochs), losses_train_dn[1:], color='blue', label='GD train loss')
plt.plot(range(epochs), losses_train[1:], color='red', label='Newton train loss')
plt.xlabel('epoch')
plt.legend()  # creates legend in top right corner of plot
plt.ylabel('train loss')
plt.title('Plot of train loss vs epoch')
plt.savefig('qf_train_loss.png', dpi=400)
plt.show()
