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
    return U, S


def project(X, U, k):
    U_reduce = U[:, :k]  # should be n x k

    (m, n) = X.shape
    assert U_reduce.shape[0] == n

    Z = U_reduce.T @ X.T
    return Z.T


def reconstruct(Z, U, k):
    U_reduce = U[:, :k]
    X_approx = U_reduce @ Z.T  # (n x k) @ (m x k)'
    return X_approx.T
