import numpy as np


def train(x_m, y):
    pass


def logistic_regression_hypothesis(x_m, theta):
    """

    :param x_m: matrix of size m x n
    :param theta: vector of length m (m x 1)
    :return:
    """
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    z = x_m @ theta

    assert z.shape == (m, 1)

    return 1 / (1 + (np.e ** -z))


def logistic_regression_cost(x_m, y, theta, h_func=logistic_regression_hypothesis):
    h = h_func(x_m, theta)
    costs = -y * np.log(h) - (1 - y) * np.log(1 - h)
    return costs.sum() / x_m.shape[0]


def logistic_regression_cost_derivative(x_m, y, theta, h_fun=logistic_regression_hypothesis):
    pass


def gradient_descent(x_m, y, cost_func, cost_func_derivative, alpha=0.1, num_iter=100):
    (m, n) = x_m.shape
    theta = np.zeros((n, 1))
    for i in range(num_iter):
        theta = theta - alpha * (cost_func_derivative(x_m, y, theta).sum() / m)
        print(cost_func(x_m, y, theta))
