import numpy as np


def gaussian(X, mu, sigma2):
    (m, n) = X.shape
    assert mu.shape == (n, 1)
    p = np.ones((m, 1))
    for i, x in enumerate(X):
        for j in range(len(x)):
            p[i, 0] *= (1 / np.sqrt(2 * np.pi * sigma2[j])) * np.exp(- ((x[j] - mu[j]) ** 2) / (2 * sigma2[j]))

    return p


def estimate_gaussian(X):
    mu = X.mean(0)
    sigma2 = X.var(0)
    return mu.reshape((-1, 1)), sigma2.reshape((-1, 1))


def select_threshold(p_cv, y_cv):
    assert p_cv.shape == y_cv.shape
    best_f1 = 0
    best_epsilon = None
    for epsilon in np.arange(min(p_cv), max(p_cv), (max(p_cv) - min(p_cv)) / 1000):
        # tp = fp = fn = 0
        # for i in range(len(p_cv)):
        #     if y_cv[i] == 1 and p_cv[i] < epsilon:
        #         tp += 1
        #     elif y_cv[i] == 0 and p_cv[i] < epsilon:
        #         fp += 1
        #     elif y_cv[i] == 1 and p_cv[i] >= epsilon:
        #         fn += 1
        tp = np.sum(np.logical_and(p_cv < epsilon, y_cv == 1))
        fp = np.sum(np.logical_and(p_cv < epsilon, y_cv == 0))
        fn = np.sum(np.logical_and(p_cv >= epsilon, y_cv == 1))
        if (tp + fp) == 0 or (tp + fn) == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
        if f1 > best_f1:
            best_epsilon = epsilon
            best_f1 = f1
    return best_epsilon, best_f1