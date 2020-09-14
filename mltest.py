import unittest

import numpy as np

import ml.feature as feature
import ml.ml as ml


class TestMlMethods(unittest.TestCase):

    def assert_equal_ndarrays(self, a1, a2):
        self.assertTrue((a1 == a2).all(), "\n{}\nis different from\n{}".format(a1, a2))

    def test_logistic_regression_hypothesis(self):
        theta = np.array([[1, 2]]).T
        x = np.array([[1, 1], [1, 2], [1, 1]])
        y = np.array([[1, 0, 1]]).T
        print(x)
        print(ml.logistic_regression_cost(x, y, theta))

    def test_la_stuff(self):
        v = np.array([[1, 3, 2]]).T

        self.assert_equal_ndarrays(v * 2, np.array([[2, 6, 4]]).T)

        self.assert_equal_ndarrays(np.log10(np.array([[1, 10, 100]]).T), np.array([[0, 1, 2]]).T)

    def test_feature_one_hot_encoding(self):
        X = np.array([
            [1, 2],
            [1, 3],
            [1, 1]])
        X_expected = np.array([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
        ])
        self.assert_equal_ndarrays(X_expected, feature.one_hot_encode(X, 1))

    def test_reduce_features_without_std(self):
        X = np.array([
            [2, 2, 1, 3],
            [2, 3, 1, 0]
        ])
        X_expected = np.array([
            [2, 3],
            [3, 0]
        ])
        self.assert_equal_ndarrays(X_expected, feature.reduce_features_without_std(X))

if __name__ == '__main__':
    unittest.main()
