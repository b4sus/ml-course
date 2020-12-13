import numpy as np


def pca(X):
    """
    Computes pca
    :param X: should be normalized
    :return:
    """
    (m, n) = X.shape
    Sigma = X.T @ X / m
    (U, S, V) = np.linalg.svd(Sigma)
    return U[:, 0]
