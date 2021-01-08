import unittest

import numpy as np

import ml.feature as feature
import ml.logistic_regression as lori


class TestMlMethods(unittest.TestCase):

    def assert_equal_ndarrays(self, a1, a2):
        self.assertEqual(a1.shape, a2.shape)
        self.assertTrue((a1 == a2).all(), "\n{}\nis different from\n{}".format(a1, a2))

    def test_logistic_regression_hypothesis(self):
        theta = np.array([[1, 2]]).T
        x = np.array([[1, 1], [1, 2], [1, 1]])
        y = np.array([[1, 0, 1]]).T
        print(x)
        print(lori.logistic_regression_cost(x, y, theta))

    def test_la_stuff(self):
        v = np.array([[1, 3, 2]]).T

        self.assert_equal_ndarrays(v * 2, np.array([[2, 6, 4]]).T)

        self.assert_equal_ndarrays(np.log10(np.array([[1, 10, 100]]).T), np.array([[0, 1, 2]]).T)

    def test_feature_one_hot_encoding(self):
        X = np.array([
            [1, 2, 1],
            [1, 3, 2],
            [1, 1, 2]])
        X_expected = np.array([
            [1,  0, 1, 0,  1, 0],
            [1,  0, 0, 1,  0, 1],
            [1,  1, 0, 0,  0, 1],
        ])
        self.assert_equal_ndarrays(X_expected, feature.one_hot_encode(X, 1, 2))

    def test_reduce_features_without_std(self):
        X = np.array([
            [2, 2, 1, 3],
            [2, 3, 1, 0]
        ])
        X_expected = np.array([
            [2, 3],
            [3, 0]
        ])

        (X_reduced, non_zero_indices) = feature.reduce_features_without_std(X)

        self.assert_equal_ndarrays(X_expected, X_reduced)
        self.assert_equal_ndarrays(np.array([1, 3]), non_zero_indices)

if __name__ == '__main__':
    unittest.main()
