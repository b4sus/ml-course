import numpy as np
import ml.ml as ml

x = np.empty((0, 3), float)
y = np.empty((0, 1), float)

with open("data/ex2data1.txt", "r") as data:
    for line in data:
        nums = line.split(",")
        x = np.vstack((x, np.array([1, float(nums[0]), float(nums[1])])))
        y = np.vstack((y, np.array([float(nums[2])])))

theta = ml.gradient_descent(x, y, ml.logistic_regression_cost, ml.logistic_regression_cost_derivative)

print(theta)