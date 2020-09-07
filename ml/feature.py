import numpy as np


class FeatureNormalizer(object):

    def __init__(self, x_m):
        (self.normalized_x_m, self.means, self.stds) = FeatureNormalizer.__normalize(x_m)

    def normalize(self, x):
        return (x - self.means) / self.stds

    @staticmethod
    def __normalize(x_m):
        (m, n) = x_m.shape
        normalized_x_m = np.empty((m, 0), float)
        means = np.empty(n)
        stds = np.empty(n)
        for feature_idx in range(n):
            feature = x_m[:, feature_idx]
            means[feature_idx] = feature.mean()
            stds[feature_idx] = feature.std()
            normalized_feature = ((feature - means[feature_idx]) / stds[feature_idx]).T.reshape((m, 1))
            normalized_x_m = np.hstack((normalized_x_m, normalized_feature))
        return normalized_x_m, means, stds


def make_polynomial(x_m, degree=6):
    (m, n) = x_m.shape
    assert n == 2
    new_x_m = np.empty([0, 0])
    for i in range(m):
        poly_feature = __make_polynomial(x_m[i, 0], x_m[i, 1], degree)
        new_x_m = np.vstack((new_x_m, poly_feature))
    return new_x_m


def __make_polynomial(x1, x2, degree):
    x = np.array([1])
    for i in range(1, degree):
        for j in range(i):
            x = np.hstack((x, x1**(i-j) * x2**(j)))
    return x
