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

def makePolynomial(x_m, grade=6):
