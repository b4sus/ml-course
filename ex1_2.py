import numpy as np
import matplotlib.pyplot as plt
import ml.ml as ml
import ml.feature as ft

data = np.loadtxt("data/ex1data2.txt", delimiter=",")

(m, feature_length_with_result) = data.shape

x = data[:, 0:feature_length_with_result - 1].reshape((m, feature_length_with_result - 1))
y = data[:, feature_length_with_result - 1].reshape((m, 1))

normalizer = ft.FeatureNormalizer(x)

normalized_x_m = np.hstack((np.ones((m, 1)), normalizer.normalized_x_m))

(theta, costs) = ml.gradient_descent(normalized_x_m, y, ml.linear_regression_cost, ml.linear_regression_cost_derivative)

plt.plot(costs)
plt.show()

example = np.array([2400, 6])

normalized_example = normalizer.normalize(example)

estimation = np.hstack((np.array([1]), normalized_example)) @ theta

print(estimation)
