import numpy as np


def train(x_m, y):
    pass


def logistic_regression_hypothesis(x, theta):
    z = theta.transpose() * x
    return 1 / (1 - np.e ^ z)
