import scipy.io as sio
import matplotlib.pyplot as plt
import ml.ml as ml
import numpy as np

data = sio.loadmat("ml_course_solutions/machine-learning-ex5/ex5/ex5data1.mat")

X_train = data["X"]
y_train = data["y"]

X_cv = data["Xval"]
y_cv = data["yval"]

X_test = data["Xtest"]
y_test = data["ytest"]

plt.plot(X_train, y_train, "rx")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
plt.show()

cost = ml.linear_regression_cost(np.hstack((np.ones((len(X_train), 1)), X_train)), y_train, np.array([[1], [1]]), 1)

print(f"cost={cost}")

gradient = ml.linear_regression_cost_derivative(np.hstack((np.ones((len(X_train), 1)), X_train)), y_train, np.array([[1], [1]]), 1)

print(gradient)