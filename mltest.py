import ml.ml as ml
import unittest
import numpy as np


class TestMlMethods(unittest.TestCase):

    def assert_equal_ndarrays(self, a1, a2):
        self.assertTrue((a1 == a2).all())

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


if __name__ == '__main__':
    unittest.main()
