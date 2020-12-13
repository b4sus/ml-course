import numpy as np


class FeatureNormalizer(object):

    def __init__(self, x_m):
        (self.normalized_x_m, self.means, self.stds) = FeatureNormalizer.__normalize(x_m)

    def normalize(self, x):
        return (x - self.means) / self.stds

    def normalize_matrix(self, X):
        (m, n) = X.shape

        X_norm = np.empty((0, n), float)

        for x in X:
            x_norm = self.normalize(x)
            X_norm = np.vstack((X_norm, x_norm))

        return X_norm

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
            if stds[feature_idx] == 0:
                raise Exception("Feature {} has 0 standard deviation".format(feature_idx))
            normalized_feature = ((feature - means[feature_idx]) / stds[feature_idx]).reshape((m, 1))
            normalized_x_m = np.hstack((normalized_x_m, normalized_feature))
        return normalized_x_m, means, stds


def reduce_features_without_std(X):
    """
    Sometimes making features polynomial can lead to feature with standard deviation == 0.
    Eg feature x0 is gender {-1, 1}, then x0^2 will always be 1 and that feature will not contribute
    :param X: of shape m x n
    :return: copy of input (shape m x o, where o < n) omitting the columns with std == 0
    """
    (m, n) = X.shape
    stds = np.zeros(n)
    for feature_idx in range(n):
        feature = X[:, feature_idx]
        stds[feature_idx] = feature.std()

    non_zero_indices = np.nonzero(stds)[0]
    return X[:, non_zero_indices].reshape((m, -1)), non_zero_indices


def one_hot_encode(X, feature_indices):
    def one_hot_encode(X, feature_idx):
        distinct_values = set(X[:, feature_idx])

        new_feature_map = {}

        for (idx, distinct_value) in enumerate(distinct_values):
            new_feature_map[distinct_value] = np.zeros((1, len(distinct_values)))
            new_feature_map[distinct_value][0, idx] = 1

        X_new_features = np.empty((0, len(distinct_values)))

        for row in X:
            X_new_features = np.vstack((X_new_features, new_feature_map[row[feature_idx]]))

        return np.hstack((X[:, :feature_idx], X_new_features, X[:, feature_idx + 1:])), len(distinct_values)

    feature_indices.sort()

    X_new = X

    total_new_columns = 0
    for feature_idx in feature_indices:
        feature_idx_new = feature_idx + (0 if total_new_columns == 0 else total_new_columns - 1)
        (X_new, new_columns) = one_hot_encode(X_new, feature_idx_new)
        total_new_columns += new_columns

    return X_new
