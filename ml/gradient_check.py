import numpy as np
from functools import partial
import ml


def neural_network_gradient_check():
    Theta0 = ml.initialize_random_theta((5, 3))
    Theta1 = ml.initialize_random_theta((3, 5))
    X = ml.initialize_random_theta((5, 2))
    Y = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])

    (theta, shapes) = ml.flatten_and_stack([Theta0, Theta1])
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
    logistic_regression_gradient_check()
    neural_network_gradient_check()