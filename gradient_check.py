import numpy as np
from functools import partial

import ml.ml as ml
import ml.utils as utils
import ml.collaborative_filtering as cofi


def neural_network_gradient_check():
    Theta0 = ml.initialize_random_theta((5, 3))
    Theta1 = ml.initialize_random_theta((3, 5))
    X = ml.initialize_random_theta((5, 2))
    Y = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])

    (theta, shapes) = utils.flatten_and_stack([Theta0, Theta1])
    costFunction = partial(ml.neural_network_cost_unrolled, X=X, Y=Y, shapes=shapes, regularization_lambda=1)
    numerical_gradient = compute_numerical_gradient(costFunction, theta)
    back_prop_gradient = ml.neural_network_cost_gradient_unrolled(theta, X, Y, shapes, regularization_lambda=1)[1].reshape((-1, 1))
    print(np.hstack((back_prop_gradient, numerical_gradient)))


def logistic_regression_gradient_check():
    X = np.array([[34, 78],
                  [30, 43],
                  [35, 72],
                  [60, 86],
                  [79, 75]])
    y = np.array([[0], [0], [0], [1], [1]])
    theta = np.zeros((2, 1))
    costFunction = partial(ml.logistic_regression_cost, x_m=X, y=y)
    numerical_gradient = compute_numerical_gradient(costFunction, theta)
    derivative_gradient = ml.logistic_regression_cost_derivative(X, y, theta)
    print(np.hstack((derivative_gradient, numerical_gradient)))


def collaborative_filtering_gradient_check():
    X = np.array([[1.04869, -0.40023, 1.19412],
                  [0.78085, -0.38563, 0.52120],
                  [0.64151, -0.54785, -0.08380],
                  [0.45362, -0.80022, 0.68048],
                  [0.93754, 0.10609,  0.36195]])
    Theta = np.array([[0.28544, -1.68427, 0.26294],
                      [0.50501, -0.45465, 0.31746],
                      [-0.43192, -0.47880, 0.84671],
                      [0.72860, -0.27189, 0.32684]])
    Y = np.array([[5, 4, 0, 0],
                  [3, 0, 0, 0],
                  [4, 0, 0, 0],
                  [3, 0, 0, 0],
                  [3, 0, 0, 0]])
    R = np.array([[1, 1, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0]])

    (params, shapes) = utils.flatten_and_stack([X, Theta])

    def cost_function(theta):
        Matrices = utils.roll(theta, shapes)
        return cofi.cost_function(Matrices[0], Y, R, Matrices[1])

    numerical_gradient = compute_numerical_gradient(cost_function, params)
    derivative_gradient = cofi.cost_function_gradient(params, shapes, Y, R)[1][0]

    print(np.hstack((derivative_gradient, numerical_gradient)))


def compute_numerical_gradient(costFunction, theta_vec, epsilon=0.0001):
    numerical_gradient = np.zeros(theta_vec.shape)
    perturbation = np.zeros(theta_vec.shape)
    for i in range(len(theta_vec)):
        perturbation[i, 0] = epsilon
        loss1 = costFunction(theta=theta_vec + perturbation)
        loss2 = costFunction(theta=theta_vec - perturbation)
        numerical_gradient[i, 0] = (loss1 - loss2) / (2 * epsilon)
        perturbation[i, 0] = 0
    return numerical_gradient


if __name__ == '__main__':
    # logistic_regression_gradient_check()
    # neural_network_gradient_check()
    collaborative_filtering_gradient_check()