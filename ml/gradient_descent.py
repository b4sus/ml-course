import numpy as np


def gradient_descent(x_m, y, cost_func, cost_func_derivative, alpha=0.01, num_iter=1000, regularization=0):
    (m, n) = x_m.shape
    theta = np.zeros((n, 1))
    costs = np.empty((num_iter, 1))
    for i in range(num_iter):
        costs[i, 0] = cost_func(x_m, y, theta)
        der = cost_func_derivative(x_m, y, theta, regularization)
        theta = theta - alpha * der
    return theta, costs