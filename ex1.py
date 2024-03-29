import numpy as np
import matplotlib.pyplot as plt

import ml.linear_regression as lire
from ml.gradient_descent import gradient_descent

data = np.loadtxt("ml_course_material/machine-learning-ex1/ex1/ex1data1.txt", delimiter=",")

(m, feature_length_with_result) = data.shape

x = data[:, 0].reshape((m, feature_length_with_result - 1))
y = data[:, 1].reshape((m, 1))

x_m = np.hstack((np.ones((m, 1)), x))

(theta, costs) = gradient_descent(x_m, y, lire.linear_regression_cost, lire.linear_regression_cost_derivative)

plt.subplot(211)
plt.plot(x, y, "xr", x, x_m @ theta)
plt.subplot(212)
plt.plot(costs)
plt.show()
