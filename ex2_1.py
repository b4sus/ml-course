import matplotlib.pyplot as plt
import numpy as np

import ml.feature as ft
import ml.ml as ml
import ml.predict as predict

data = np.loadtxt("data/ex2data1.txt", delimiter=",")

(m, feature_length_with_result) = data.shape

x = data[:, 0:feature_length_with_result - 1].reshape((m, feature_length_with_result - 1))
y = data[:, feature_length_with_result - 1].reshape((m, 1))

success_indices = np.nonzero(y.flatten() == 1)
fail_indices = np.nonzero(y.flatten() == 0)

normalizer = ft.FeatureNormalizer(x)

normalized_x_m = np.hstack((np.ones((m, 1)), normalizer.normalized_x_m))

(theta, costs) = ml.gradient_descent(normalized_x_m, y, ml.logistic_regression_cost,
                                     ml.logistic_regression_cost_derivative, alpha=0.01, num_iter=10000)

plt.figure(0)
plt.subplot(211)
plt.plot(x[success_indices, 0], x[success_indices, 1], "kx")
plt.plot(x[fail_indices, 0], x[fail_indices, 1], "yo")
plt.plot()
plt.subplot(212)
plt.plot(costs)
plt.show()

normalized_example = np.hstack((np.array([1]), normalizer.normalize(np.array([45, 80])))).reshape((1, 3))

print(ml.logistic_regression_hypothesis(normalized_example, theta))

predictions = predict.predict(x, theta, normalizer, ml.logistic_regression_hypothesis)

print(np.mean(predictions == y))

space_x = np.linspace(0, 100, num=100)
space_y = np.linspace(0, 100, num=100)

Z = np.zeros((len(space_x), len(space_y)))
for i in range(len(space_x)):
    for j in range(len(space_y)):
        Z[i, j] = ml.logistic_regression_hypothesis(
            np.hstack((np.array([1]), normalizer.normalize(np.array([space_x[i], space_y[j]])))).reshape((1, 3)), theta)
        # Z[i, j] = predict.predict(np.array([[space_x[i], space_y[j]]]), theta, normalizer, ml.logistic_regression_hypothesis)

plt.figure(1)
plt.contourf(space_x, space_y, Z)
plt.show()
