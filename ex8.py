import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

import ml.pca as pca
import ml.feature as feature


def visualize(X, figure=0, *, block=False):
    plt.figure(figure)
    plt.plot(X[:, 0], X[:, 1], "x")
    plt.show(block=block)


def visualize_fit(X, mu, sigma2, *, block=False):
    ls = np.linspace(0, 30, 100)
    (mesh_x, mesh_y) = np.meshgrid(ls, ls)
    Z = gaussian(np.c_[mesh_x.flatten(), mesh_y.flatten()], mu, sigma2)
    plt.contour(mesh_x, mesh_y, Z.reshape((100, 100)), [10 ** i for i in range(-20, 0, 3)])
    plt.show(block=block)


def gaussian(X, mu, sigma2):
    (m, n) = X.shape
    assert mu.shape == (n, 1)
    p = np.ones((m, 1))
    for i, x in enumerate(X):
        for j in range(len(x)):
            p[i, 0] *= (1 / np.sqrt(2 * np.pi * sigma2[j])) * np.exp(- ((x[j] - mu[j]) ** 2) / (2 * sigma2[j]))

    return p


def estimate_gaussian(X):
    mu = X.mean(0)
    sigma2 = X.var(0)
    return mu.reshape((-1, 1)), sigma2.reshape((-1, 1))


def select_threshold(p_cv, y_cv):
    assert p_cv.shape == y_cv.shape
    best_f1 = 0
    best_epsilon = None
    for epsilon in np.arange(min(p_cv), max(p_cv), (max(p_cv) - min(p_cv)) / 1000):
        # tp = fp = fn = 0
        # for i in range(len(p_cv)):
        #     if y_cv[i] == 1 and p_cv[i] < epsilon:
        #         tp += 1
        #     elif y_cv[i] == 0 and p_cv[i] < epsilon:
        #         fp += 1
        #     elif y_cv[i] == 1 and p_cv[i] >= epsilon:
        #         fn += 1
        tp = np.sum(np.logical_and(p_cv < epsilon, y_cv == 1))
        fp = np.sum(np.logical_and(p_cv < epsilon, y_cv == 0))
        fn = np.sum(np.logical_and(p_cv >= epsilon, y_cv == 1))
        if (tp + fp) == 0 or (tp + fn) == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
        if f1 > best_f1:
            best_epsilon = epsilon
            best_f1 = f1
    return best_epsilon, best_f1


def server_anomaly_detection():
    data = sio.loadmat("ml_course_material/machine-learning-ex8/ex8/ex8data1.mat")
    X = data["X"]
    X_cv = data["Xval"]
    y_cv = data["yval"]

    visualize(X, block=False)
    (mu, sigma2) = estimate_gaussian(X)
    visualize_fit(X, mu, sigma2, block=False)
    p_cv = gaussian(X_cv, mu, sigma2)
    (epsilon, f1) = select_threshold(p_cv, y_cv)
    print(f"best epsilon is {epsilon} with F1 score {f1}")
    p = gaussian(X, mu, sigma2)
    (anomalies_indices_x, anomalies_indices_y) = np.nonzero(p < epsilon)
    plt.plot(X[anomalies_indices_x, 0], X[anomalies_indices_x, 1], '+')
    plt.show()


def anomaly_detection_high_dimensional_dataset():
    data = sio.loadmat("ml_course_material/machine-learning-ex8/ex8/ex8data2.mat")
    X = data["X"]
    X_cv = data["Xval"]
    y_cv = data["yval"]

    (mu, sigma2) = estimate_gaussian(X)
    p_cv = gaussian(X_cv, mu, sigma2)
    (epsilon, f1) = select_threshold(p_cv, y_cv)
    print(f"best epsilon is {epsilon} with F1 score {f1}")
    p = gaussian(X, mu, sigma2)
    (anomalies_indices_x, anomalies_indices_y) = np.nonzero(p < epsilon)
    print(f"{len(anomalies_indices_x)} anomalies")

    # pca for fun
    normnalizer = feature.FeatureNormalizer(X)
    X_norm = normnalizer.normalized_x_m
    (U, S) = pca.pca(X_norm)
    Z = pca.project(X_norm, U, 2)
    visualize(Z)
    plt.plot(Z[anomalies_indices_x, 0], Z[anomalies_indices_x, 1], '+')
    plt.show()


if __name__ == "__main__":
    anomaly_detection_high_dimensional_dataset()
