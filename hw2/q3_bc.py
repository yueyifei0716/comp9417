import numpy as np
import matplotlib.pyplot as plt

# Question b of Part 3
sigma = 1

bias_MLE = lambda n: (-1 * sigma ** 2) / n
var_MLE = lambda n: (2 * sigma ** 4 * (n - 1)) / n ** 2

bias_BE = lambda n: n * 0
var_BE = lambda n: (2 * sigma ** 4) / (n - 1)

nrange = np.linspace(1, 250, 1000)

# Plot the bias of both estimators
plt.plot(nrange, bias_MLE(nrange), label="MLE Estimator", color='blue')
plt.plot(nrange, bias_BE(nrange), label="Question b Estimator", color="red")

plt.xlabel('Sample Size n')
plt.ylabel('Bias')
plt.title('The bias of both estimators')
plt.legend()
plt.savefig('q3b_bias.png', dpi=400)
plt.show()

# the variance of both estimators
plt.plot(nrange, var_MLE(nrange), label="MLE Estimator", color='blue')
plt.plot(nrange, var_BE(nrange), label="Question b Estimator", color="red")

plt.xlabel('Sample Size n')
plt.ylabel('Variance')
plt.title('The variance of both estimators')
plt.legend()
plt.savefig('q3b_var.png', dpi=400)
plt.show()

# Question c of Part 3
MSE_MLE = lambda n: bias_MLE(n) ** 2 + var_MLE(n)
MSE_BE = lambda n: bias_BE(n) ** 2 + var_BE(n)

# the MSEs of both estimators
plt.plot(nrange, MSE_MLE(nrange), label="MLE Estimator", color='blue')
plt.plot(nrange, MSE_BE(nrange), label="Question b Estimator", color="red")

plt.xlabel('Sample Size n')
plt.ylabel('MSE')
plt.title('The MSEs of both estimators')
plt.legend()
plt.savefig('q3c.png', dpi=400)
plt.show()
