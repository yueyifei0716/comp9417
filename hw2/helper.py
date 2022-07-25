import numpy as np
from sklearn.metrics import log_loss

# these are helper functions, you don't have to use them if you would like to write your own code

def sigmoid(x):
    # logistic sigmoid
    return np.exp(-np.logaddexp(0, -x))

def loss(gamma, X, y, lam):
    # gamma has first coordinate = beta0 = intercept, and second coordinate = beta
    norm_beta_sq = np.linalg.norm(gamma[1:], ord=2)**2
    z = np.dot(X, gamma[1:]) + gamma[0]
    sig_z = sigmoid(z)
    return lam * log_loss(y, sig_z, normalize=True) + 0.5 * norm_beta_sq
