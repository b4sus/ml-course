import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def visualize(X, figure=0, block=False):
    plt.figure(figure)
    plt.plot(X[:, 0], X[:, 1], "x")
    plt.show(block=block)


def visualize_fit(X, mu, sigma2):
    ls = np.linspace(0, 30, 100)
    (mesh_x, mesh_y) = np.meshgrid(ls, ls)
    Z = gaussian(np.c_[mesh_x.flatten(), mesh_y.flatten()], mu, sigma2)
    plt.contour(mesh_x, mesh_y, Z.reshape((100, 100)), [10 ** i for i in range(-20, 0, 3)])
    plt.show()


def gaussian(X, mu, sigma2):
    (m, n) = X.shape
    assert mu.shape == (n, 1)
    p = np.ones(m)
    for i, x in enumerate(X):
        for j in range(len(x)):
            p[i] *= (1 / np.sqrt(2 * np.pi * sigma2[j])) * np.exp(- ((x[j] - mu[j]) ** 2) / (2 * sigma2[j]))

    return p


def estimate_gaussian(X):
    mu = X.mean(0)
    sigma2 = X.var(0)
    return mu.reshape((-1, 1)), sigma2.reshape((-1, 1))


if __name__ == "__main__":
    data = sio.loadmat("ml_course_solutions/machine-learning-ex8/ex8/ex8data1.mat")
    X = data["X"]
    X_val = data["Xval"]
    y_val = data["yval"]

    visualize(X, block=False)
    (mu, sigma2) = estimate_gaussian(X)
    visualize_fit(X, mu, sigma2)